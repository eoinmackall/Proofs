"""Multi-agent proof loop: planner -> selector -> prover -> verifier -> reviser -> DAG.

Design note on thinking vs. structured output
---------------------------------------------
llama.cpp's `json_schema` response_format is *ignored* while reasoning is on,
so the model is free to wrap its JSON in a markdown fence, and the
server's own JSON parser then 500s (ggml-org/llama.cpp#20345). The fix has
two halves:

  * `llm_backend.py` neutralizes the conflict per call: thinking is turned
    off for exactly the calls that carry a schema (the extraction stage),
    and left on everywhere else.
  * `extract()` additionally degrades to prompt-only with a JSON repair
    pipeline in `clean_json_text()`, in case the model still wraps the
    output in a fence or emits stray prose.

The reasoning stage stays schema-free, because that is exactly where the
model needs to think.

All seven agent prompts (planner.md, selector.md, prover.md, verifier_1.md,
verifier_2.md, verifier_3.md and reviser.md) instruct the model to emit
strict JSON directly, so each
response is first
parsed as-is; the extraction stage only runs as a fallback when the model
fails to comply. The prover never round-trips through a second model call at
all: its JSON is parsed locally and, failing that, the reasoning content is
taken verbatim, so the proof text can't be abridged or paraphrased.

Planning is a shortlist, not a decision
---------------------------------------
The planner proposes PLANNER_CANDIDATES (5) lemmas per iteration rather than
one, ordered best-first, and something downstream picks exactly one. In
--mode auto that picker is the selector agent: screen_candidates() filters
the shortlist against the DAG, and when more than one candidate survives,
selector.md is asked to weigh them all and name the one it judges most
likely to come through the prover and the verifiers. The planner's best-first
order is the fallback, not the default — it is what gets taken when the
selector fails, names a candidate it was not offered, or when only one
candidate survives, where there is nothing left to choose. A candidate naming
an unproved dependency still costs a candidate rather than a whole iteration.

Proving is a loop, not a single pass
------------------------------------
A lemma enters the DAG only after all three verifier agents have accepted the
same proof. Verification is three atomic steps rather than one: verifier_1,
verifier_2 and verifier_3 are separate agents — separate prompt files and
separate roles — each a single call, so each step can later grow its
own focus (for now all three prompts are identical). Any single reject ends
the counting and sends the proof to the reviser with the
verifier's reasoning about the failure. The reviser
decides where the fault lies: a fault in the statement comes back as a
revised statement, which becomes the prover's new target; a fault in the
argument keeps the statement and sends the prover back with the verdict as
feedback. The loop allows MAX_PROOF_ATTEMPTS prover rounds per lemma per
iteration (--max-proof-attempts), then gives the lemma up and asks the
planner again.

When a call runs into the context wall
--------------------------------------
The server is stateless, so a truncated exchange — prompt, thinking trace,
answer-so-far, summing to about num_ctx — cannot be re-sent whole. reason()
therefore gets pi-style compaction rescues before reporting "ceiling":
the thinking trace (scratch, regenerable) is summarised into a structured
summary — Goal, Progress, Key Decisions, Next Steps, Critical Context, plus
a <cited-lemmas> tracking block — in chunked passes if it is big; the task
prompt and the answer written so far are kept verbatim, the answer ending
exactly where it was cut; and the model is asked to resume from the cut with
the whole headroom of the now-smaller prompt. The passes are unbounded: a
continuation that hits the wall itself becomes the next pass's input, its
trace folded into the summary, its text extending the verbatim answer, so
the loop runs as long as the answer keeps advancing. Only a pass that leaves
nothing to build on, or no headroom left, degrades to the "ceiling" the
planner/prover/verifier callers already handle.

In --mode human the picker is a person, and picking is two decisions rather
than one. A number and Enter accepts that candidate as true on your
authority: it goes straight into the DAG, with no prover call and no verifier
calls, marked "provenance": "operator". A trailing 'p' — "3p" — sends it down
the ordinary pipeline instead, for when you are confident it is the right next
step but not that it is true. You can also write your own lemma ('w', or 'wp'
to have it proved), or send the planner back to think again.

You can cross between them mid-run, in either direction, without restarting.
'h' toggles: press it during a planner, prover, verifier or reviser call and
the mode flips at the next planning step, which is the earliest a change of
mode can mean anything. 'a' at the menu does the same thing for the case where the
menu is already up and the listener is therefore paused. See interaction.py
for why one of these is a keystroke and the other is a line of input.

Usage
-----
    python main.py --conjecture algebra_example
    python main.py --conjecture algebra_example --model Qwen3.5-122B-Q4_K_M
    python main.py --conjecture algebra_example --model qwen3.8-27b
    python main.py --conjecture algebra_example --no-verbose --max-iterations 25
    python main.py --conjecture algebra_example --mode human
    python parsing.py --conjecture algebra_example

parsing.py is standalone: it reads conjectures/<name>/references/ (any .tex
or .md file in it), asks the model for the theorem-level results, and writes
them as a strict-JSON array to conjectures/<name>/references.md. main.py then
offers those references to the prover (citable by id), to the verifiers (the
formal statements of the cited ones only) and to the planner (an id + slogan
shortlist, only while the collection is small enough to be one).

--conjecture names a directory under conjectures/ holding a conjecture.md.
The DAG is written beside it as dag.json and shared by every model: point a
second model at a conjecture already under way and it continues from the
lemmas the first one proved. Pass --dag to give a run its own file instead.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

import interaction
import llm_backend
import workspace

# ----------------------------------------------------------------------------
# Configuration (all overridable from the command line; see main())
# ----------------------------------------------------------------------------
# qwen3.8-27b — llama-server reached over the SSH tunnel on 8081
# (the only live inference server on this machine).
MODEL_NAME = "qwen3.8-27b"
DEFAULT_CONJECTURE = "algebra_example"

# Set in main(). BACKEND owns the HTTP conversation; PROFILE is what the
# capability probe found (context ceiling, thinking support, tokenizer ratio).
BACKEND: Any = None
PROFILE: Any = None

# Resolved by workspace.resolve(): a prompt may be overridden per conjecture,
# so these are paths rather than the bare filenames they used to be.
CONJECTURE_FILE = ""
DAG_FILE = ""
REFERENCES_FILE = ""
PROMPT_PATHS: Dict[str, str] = {name: name for name in workspace.PROMPT_FILES}

MAX_ITERATIONS = 10
LLM_MAX_RETRIES = 3
REQUEST_TIMEOUT = 1800       # Thinking models are slow; give them room

# How many lemmas the planner shortlists per iteration. Raising this costs
# planner tokens and, in human mode, attention; five is about as many
# candidates as can be compared without re-reading the conjecture.
PLANNER_CANDIDATES = 5

# How many parsed references (references.md) the planner is shown at all. The
# rule is deliberately blunt: below the limit the planner gets an id + slogan
# per result, so it can build on a named theorem rather than re-prove it from
# scratch; at or above it, none. A shortlist of hundreds of slogans costs
# more in tokens and attention than it returns, and the prover sees the full
# collection regardless, so a named result the planner misses is one proving
# round away, not lost. A strategy for large collections is deliberately not
# built yet.
PLANNER_REFERENCE_LIMIT = 100

# The verification steps: a proof enters the DAG only after every one of
# these verifier agents has accepted it. Each is a separate agent — its own
# prompt file and role, run as a single atomic check — rather than one
# verifier run repeatedly, so each step can grow its own focus. For now all
# three prompts are identical. One reject ends the counting and sends the
# proof to the reviser.
VERIFIER_AGENTS = ("verifier_1.md", "verifier_2.md", "verifier_3.md")
# How many prover rounds one iteration may spend on a lemma before giving it
# up and re-planning.
MAX_PROOF_ATTEMPTS = 12

# "auto" reproduces the original behaviour end to end. "human" stops at every
# planning step. Set from --mode, then mutated by the hotkey and the menu, so
# it is genuinely a run-time toggle rather than a launch-time one.
DEFAULT_MODE = "auto"
MODE = DEFAULT_MODE
HOTKEY_KEYS = "h"            # single keystroke; see interaction.HotKey
HOTKEY: interaction.HotKey = interaction.NullHotKey()

# Default context window. main() reassigns it after probing: with no
# --num-ctx the server's own -c is used as-is (the KV cache is already
# allocated at launch, so there is nothing to request); an explicit
# --num-ctx always wins, clamped to the probed ceiling. This constant stands
# in when the probe fails. 65536 matches the Qwen3.8-27B reference launch
# line in llm_backend.py (-c 65536).
NUM_CTX = 65536
# No static per-role generation budgets. reason() and extract() give every
# call the full headroom — num_ctx minus the prompt, minus a safety margin —
# because num_predict is a ceiling, not a target: the model stops at EOS
# whatever the number, so a tight budget can only ever end in a truncated
# generation paid for in full, followed by a retry with a bigger number. The
# response is allowed to use the whole window, which means a truncation now
# reports "ceiling" — the lemma is too big for the context — instead of
# quietly undersizing the answer.

# Thinking level: True, or "low"/"medium"/"high"/"max" on models that support
# levels. Set to None to omit the field entirely (the model's default).
# Never set this to False — see the module docstring.
#
# These are requests, not guarantees. The backend checks them against the
# probed capabilities and drops or downgrades: asking a non-thinking model to
# think is a 400 from the server, and a level string sent to a model that
# only understands booleans is ignored silently, which is worse.
THINK = {
    "planner": True,
    "selector": True,
    "prover": True,
    "verifier_1": True,
    "verifier_2": True,
    "verifier_3": True,
    "reviser": True,
    "parser": True,   # parsing.py: reading mathematics, then extracting it
}

# Sampling. The presence penalty is the delicate one: it pushes the model off
# tokens it has already used, which suppresses repetition loops in agentic
# coding but is actively harmful here — mathematics *requires* hammering the
# same symbols (\epsilon, n, x_i) over and over, and the extraction stage's
# whole job is verbatim copying.
#
# The right value is model-specific, so the per-model baseline lives in
# llm_backend.OVERRIDES and the backend layers these on top; anything set
# here wins, anything omitted takes the model's recommended value. Keep the
# dict minimal for that reason.
REASONING_OPTIONS: Dict[str, Any] = {
    "temperature": 0.7,          # Greedy decoding degrades thinking models.
    "num_ctx": NUM_CTX,
    # presence_penalty: deliberately absent — see llm_backend.OVERRIDES.
    # num_predict is set per call by reason(); see the NUM_CTX comment.
}

EXTRACT_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,          # Mechanical, and grammar-constrained anyway.
    "presence_penalty": 0.0,     # Must be 0 on every model: this stage copies,
                                 # it doesn't write. Stated explicitly so it
                                 # overrides whatever OVERRIDES recommends.
    "min_p": 0.0,
    "num_ctx": NUM_CTX,
    # num_predict is set per call by extract(); see the NUM_CTX comment.
}

# Per-role reasoning temperature overrides.
TEMPERATURES = {
    "planner": 0.7,
    "selector": 0.5,   # Comparing a shortlist, not generating one.
    "prover": 0.6,     # Slightly tighter: rigour over exploration.
    "verifier_1": 0.8,
    "verifier_2": 0.8,
    "verifier_3": 0.8,   # Looser, so the three independent checks don't
                        # collapse into one review of the first pass.
    "reviser": 0.7,    # Diagnosing a failure is planning-scale judgement.
    "parser": 0.3,     # Extracting from a fixed source, not exploring.
}

# Pi-style compaction on the context wall: a truncated call is resumed by
# summarising its thinking trace and re-sending task + answer-so-far (see
# _resume_compacted, between _headroom and reason below). A truncated
# continuation is not a failure but the next pass, so the passes run until
# the answer completes (or nothing is left to build on). COMPACT_CHUNK_TOKENS
# bounds one compaction pass's input, COMPACT_TAIL_CHARS is what the compactor
# sees of the answer-so-far, COMPACT_MIN_ROOM is the smallest headroom a
# continuation is worth attempting.
COMPACT_ENABLED = True
COMPACT_CHUNK_TOKENS = 24000
COMPACT_TAIL_CHARS = 2000
COMPACT_MIN_ROOM = 2048

# Set at runtime if schema-constrained extraction proves incompatible with the
# model's default thinking (empty content, output stranded in .thinking).
_SCHEMA_MODE_BROKEN = False


def log(message: str, verbose: bool = True) -> None:
    """Single logging chokepoint, so --no-verbose silences everything."""
    if verbose:
        print(message)


# ----------------------------------------------------------------------------
# Response schemas (llama.cpp builds a grammar from these, so required keys
# and enum values are guaranteed rather than hoped for).
# ----------------------------------------------------------------------------
# No dependencies field: the planner proposes statements, and which proved
# lemmas a proof leans on is settled by the prover while it writes the proof.
_LEMMA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "statement": {"type": "string"},
        # Which end of the question the lemma works toward. planner.md writes
        # it and planner_candidates() defaults a missing one to "proof", so
        # an old per-conjecture override that predates aims keeps working.
        "aim": {"type": "string", "enum": ["proof", "counterexample"]},
    },
    "required": ["id", "statement", "aim"],
}

# minItems/maxItems are deliberately absent: llama.cpp's schema support does
# not cover array cardinality, so writing them here would look like an
# enforced guarantee while enforcing nothing. The count is requested in
# planner.md and trimmed in planner_candidates().
PLANNER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_conjecture_proved": {"type": "boolean"},
        "is_conjecture_disproved": {"type": "boolean"},
        "plan_summary": {"type": "string"},
        "candidate_lemmas": {"type": "array", "items": _LEMMA_SCHEMA},
    },
    "required": [
        "is_conjecture_proved",
        "is_conjecture_disproved",
        "plan_summary",
        "candidate_lemmas",
    ],
}

# The selector picks by id, but "one of these particular ids" is not
# something a grammar can enforce, so the offered ids are listed in the
# prompt and the answer is checked against them in select_lemma(), which
# falls back to the planner's first candidate when it does not check out.
SELECTOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "selected_id": {"type": "string"},
    },
    "required": ["selected_id"],
}

VERIFIER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "reject"]},
        "justification": {"type": "string"},
    },
    "required": ["decision", "justification"],
}

# new_statement is a string, not a nullable: the reviser writes the empty
# string when it keeps the statement, which a grammar can enforce but a
# ["string", "null"] union is fiddlier to ask one for.
REVISER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "statement_revision": {"type": "boolean"},
        "diagnosis": {"type": "string"},
        "new_statement": {"type": "string"},
    },
    "required": ["statement_revision", "diagnosis", "new_statement"],
}


# ----------------------------------------------------------------------------
# File helpers
# ----------------------------------------------------------------------------
def load_file(filepath: Any) -> str:
    """Read prompt or markdown file contents. Accepts str or Path."""
    if not filepath or not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_dag() -> Dict[str, Any]:
    """Load current DAG state from file. Absent file => empty DAG."""
    if os.path.exists(DAG_FILE):
        with open(DAG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"lemmas": {}}


def save_dag(dag: Dict[str, Any]) -> None:
    """Save updated DAG state to file, creating it if needed."""
    with open(DAG_FILE, "w", encoding="utf-8") as f:
        json.dump(dag, f, indent=2)


# ----------------------------------------------------------------------------
# JSON repair (fallback path; the schema grammar should make it unnecessary)
# ----------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
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
    """Close an unterminated string and any open braces, so a cut-off reply is
    still usable instead of lost."""
    in_str, esc, depth = False, False, 0
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
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    repaired = text.rstrip()
    if in_str:
        repaired = repaired.rstrip("\\") + '"'
    return repaired + "}" * max(depth, 0)


def clean_json_text(raw: str) -> str:
    """Normalise model output into parseable JSON.

    Repairs escalate and each is tried only if the previous stage fails to
    parse — text that is already valid JSON is returned byte-for-byte
    untouched, so the repair heuristics can never corrupt a correct reply.
    """
    text = _THINK_RE.sub("", raw)
    if "<think>" in text:
        text = _OPEN_THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text).strip()
    text = _extract_json_object(text)

    candidates = [text]
    repaired = _repair_escapes(text)
    if repaired != text:
        candidates.append(repaired)
    candidates.append(_salvage_truncated(repaired))

    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return candidates[-1]  # let the caller surface the real parse error


def parse_json_or_none(text: str) -> Optional[Dict[str, Any]]:
    """Parse an agent reply that already complies with its prompt file.

    planner.md, selector.md, prover.md, the three verifier prompts
    (verifier_1.md, verifier_2.md, verifier_3.md) and reviser.md all end
    with "Output strictly valid JSON ... no markdown fences and no extra
    text", so the reasoning stage's content is usually the JSON object itself. When it
    parses, we use it directly and skip the extraction call entirely.
    """
    if not text:
        return None
    try:
        obj = json.loads(clean_json_text(text))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ----------------------------------------------------------------------------
# LLM plumbing
# ----------------------------------------------------------------------------
def _headroom(messages: List[Dict[str, str]], num_ctx: int) -> int:
    """Tokens available for generation after the prompt, with a safety margin.

    num_ctx caps prompt + generation together, so this is the most a response
    can ever get: anything more is cut off by the context window, not the
    budget, after a long generation.

    The prompt size used to be estimated as chars // 4. That is optimistic for
    LaTeX-dense text and, worse, it is tokenizer-specific — the whole point of
    this project is now to run the same conjecture through four different
    tokenizers. PROFILE.estimate_tokens() uses a ratio calibrated from the
    prompt_eval_count of calls already made, so the estimate converges on the
    truth for whichever model is loaded.
    """
    chars = sum(len(m["content"]) for m in messages)
    ratio = PROFILE.chars_per_token if PROFILE else 4.0
    used = int(chars / max(ratio, 1.0)) + 1
    return max(num_ctx - used - 512, 0)


COMPACT_SYSTEM = (
    "You are a compaction pass, not the agent. You are given part of a "
    "mathematician's reasoning trace that was cut off by the context limit. "
    "Write ONLY the structured summary in the format below — do not continue "
    "the mathematics, do not add claims, no preamble, no fences. If a previous "
    "pass's summary is included, fold it in: the result must cover everything "
    "it covered, plus this part.\n\n"
    "Format (fill every section, 'none' where nothing applies):\n"
    "## Goal\n[the task the trace was working on, in your own words]\n"
    "## Progress\n### Done\n- [x] ...\n### In Progress\n- [ ] ...\n"
    "### Blocked\n- ...\n## Key Decisions\n- **[decision]**: rationale\n"
    "## Next Steps\n1. [what must happen next, starting exactly where the "
    "trace stopped]\n## Critical Context\n- [definitions, facts, partial "
    "results needed to continue]\n<cited-lemmas>\n[comma-separated lemma ids "
    "the trace cited or used, or 'none']\n</cited-lemmas>"
)


def _split_inline_thinking(content: str, thinking: str) -> Tuple[str, str]:
    """Recover (content, thinking) when the model inlined its reasoning
    instead of separating it."""
    if not thinking and "\u003cthink\u003e" in content:
        m = re.search(r"\u003cthink\u003e(.*?)\u003c/think\u003e", content, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
        content = _THINK_RE.sub("", content).strip()
    return content, thinking


def _split_trace_chunks(text: str, max_tokens: int) -> List[str]:
    """Split a trace into line-aligned chunks of at most ~max_tokens, so a
    compaction pass never needs more than a fraction of the window. Breaking
    on newlines keeps no LaTeX display cut mid-line; a chunk may still exceed
    the limit by one very long line, which the pass's headroom absorbs."""
    ratio = PROFILE.chars_per_token if PROFILE else 4.0
    max_chars = max(int(max_tokens * ratio), 1000)
    chunks: List[str] = []
    cur = ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) + 1 > max_chars:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def _compact_pass(part: str, prev: str, tail: str,
                  part_no: int, n_parts: int, verbose: bool) -> str:
    """One summarisation pass over part of the cut-off trace.

    Returns the folded summary, or `prev` unchanged on any failure: the
    compaction must never lose what it already had, and it is a rescue, not a
    new stage that can fail the call.
    """
    user = (
        f"Part {part_no} of {n_parts} of the cut-off reasoning trace:\n"
        f"[Thinking trace]:\n{part}"
    )
    if prev:
        user = (
            "The summary of all previous parts — fold it in, drop nothing "
            f"from it:\n{prev}\n\n" + user
        )
    if tail:
        user += (
            "\n\n[The answer written before the cut, its tail — the work "
            f"ends here, mid-sentence]:\n{tail}"
        )
    messages = [
        {"role": "system", "content": COMPACT_SYSTEM},
        {"role": "user", "content": user},
    ]
    options = {
        **EXTRACT_OPTIONS,
        "num_predict": _headroom(messages, EXTRACT_OPTIONS["num_ctx"]),
    }
    try:
        reply = BACKEND.chat(messages, think=None, schema=None,
                             options=options)
    except (requests.RequestException, ValueError, KeyError) as e:
        log(f"  ⚠️  compaction pass {part_no}/{n_parts} transport error: {e}",
            verbose)
        return prev
    summary = (reply.content or "").strip()
    if reply.truncated or not summary:
        log(f"  ⚠️  compaction pass {part_no}/{n_parts} was truncated or "
            f"empty; keeping the previous summary.", verbose)
        return prev
    return summary


def _compact_trace(role: str, thinking: str, tail: str, verbose: bool,
                   prev_summary: str = "") -> str:
    """Summarise a cut-off reasoning trace in pi-style passes: chunked, each
    pass folding the previous summary in. prev_summary carries an earlier
    compaction round's result across, so a trace compacted in several rounds
    (a continuation that itself hit the wall) never loses what the first
    rounds established. Returns the summary, or prev_summary unchanged when
    the trace was empty or nothing survived the passes."""
    if not thinking.strip():
        return prev_summary
    parts = _split_trace_chunks(thinking, COMPACT_CHUNK_TOKENS)
    log(f"  📦 {role} trace too big to re-send; compacting in "
        f"{len(parts)} pass(es).", verbose)
    summary = prev_summary
    for i, part in enumerate(parts, 1):
        summary = _compact_pass(part, summary, tail, i, len(parts), verbose)
    return summary


def _resume_compacted(
    role: str,
    system_prompt: str,
    user_prompt: str,
    thinking: str,
    content: str,
    think: Any,
    verbose: bool,
) -> str:
    """Pi-style overflow recovery after a truncated call.

    One pass: compact the reasoning trace (the scratch, summarised), keep
    the task prompt and the answer-so-far verbatim (the recent work), and
    ask the model to resume from the cut with the whole headroom of the
    now-smaller prompt: P + summary + C is strictly less than P + trace + C
    was.

    The passes are not bounded by a count. When a continuation hits the wall
    itself, its trace and answer-so-far become the next pass's input — the
    summary carries across rounds, so nothing established earlier is lost —
    and the model resumes from the new cut. The loop ends when the answer
    completes, when a pass leaves nothing to build on, or when the prompt
    plus the verbatim answer runs out of headroom; those return "" and fall
    through to "ceiling", which the callers already handle.
    """
    answer = content
    summary = ""
    want_think = think if (think is not None and think is not False) else None
    while True:
        tail = answer[-COMPACT_TAIL_CHARS:]
        summary = _compact_trace(role, thinking, tail, verbose, summary)

        parts = [
            user_prompt,
            "---",
            "Your previous attempt at this task was cut off by the context "
            "limit.",
        ]
        if summary:
            parts += [
                "Summary of your reasoning trace (what is established, where "
                "it stopped, what it cited):",
                summary,
            ]
        if answer.strip():
            parts += [
                "Your answer so far, verbatim, ending exactly where it was "
                "cut off:",
                answer,
                "Resume from exactly that point and finish the answer. Do "
                "not repeat anything already written.",
            ]
        else:
            parts += [
                "Your previous answer had not started: the limit was hit "
                "during reasoning. Using the summary as your notes, write "
                "the complete answer now.",
            ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        room = _headroom(messages, REASONING_OPTIONS["num_ctx"])
        if room < COMPACT_MIN_ROOM:
            log(f"  ⛔ {role} compaction left only ~{room} tokens of room; "
                f"not enough to continue.", verbose)
            return ""
        options = {
            **REASONING_OPTIONS,
            "temperature": TEMPERATURES.get(role, REASONING_OPTIONS["temperature"]),
            "num_predict": room,
        }
        try:
            reply = BACKEND.chat(messages, think=want_think, schema=None,
                                 options=options)
        except (requests.RequestException, ValueError, KeyError) as e:
            log(f"  ⚠️  {role} continuation transport error: {e}", verbose)
            return ""
        cont_content, cont_thinking = _split_inline_thinking(
            reply.content or "", reply.thinking or "")
        if reply.truncated:
            if not (cont_content.strip() or cont_thinking.strip()):
                log(f"  ⛔ {role} continuation was truncated with nothing "
                    f"to build on; the answer does not fit the window.",
                    verbose)
                return ""
            # The continuation itself hit the wall: not a failure, the next
            # pass. Its trace joins the summary, its content extends the
            # verbatim answer, and the model resumes from the new cut.
            answer += cont_content
            thinking = cont_thinking
            log(f"  🔁 {role} continuation hit the wall; the next pass "
                f"compacts its trace and resumes from the new cut.",
                verbose)
            continue
        if not cont_content.strip():
            log(f"  ⚠️  {role} continuation returned empty content.", verbose)
            return ""
        return (answer + cont_content).strip()


def reason(
    system_prompt: str,
    user_prompt: str,
    role: str,
    think: Any = True,
    verbose: bool = True,
) -> Tuple[str, str]:
    """Stage 1: free-form reasoning. No `format`, so thinking is preserved.

    Returns (content, status). status is "" on success, or "ceiling" when the
    role exhausted the context window without finishing — a signal that the
    task is too large, not that the call failed.

    Truncated output is never returned as if it were complete: `done_reason ==
    "length"` means the model was cut off mid-sentence, and the budget already
    covers the whole window. Before a truncation is reported as "ceiling" the
    exchange gets pi-style compaction rescues — the thinking trace
    summarised, the answer written so far kept verbatim, the model asked to
    resume from the cut (see _resume_compacted). The passes are unbounded:
    a continuation that is itself truncated becomes the next pass's input,
    and only a pass that leaves nothing to build on, or no headroom left,
    means the task is too large for the window.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    options: Dict[str, Any] = {
        **REASONING_OPTIONS,
        "temperature": TEMPERATURES.get(role, REASONING_OPTIONS["temperature"]),
        # As much room as the window has left, recomputed for this prompt:
        # the prompt differs every iteration and every role, and there is no
        # point asking for more than the window holds or less than it does.
        "num_predict": _headroom(messages, REASONING_OPTIONS["num_ctx"]),
    }
    # `think` is passed through as a request. The backend reconciles it with
    # the probed capabilities and never sends False, whatever we ask for.
    want_think = think if (think is not None and think is not False) else None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            reply = BACKEND.chat(messages, think=want_think, schema=None,
                                 options=options)
            content = reply.content or ""
            thinking = reply.thinking or ""
            content, thinking = _split_inline_thinking(content, thinking)

            hit_ceiling = reply.truncated

            if content.strip() and not hit_ceiling:
                return content.strip(), ""

            if hit_ceiling:
                spent = reply.eval_tokens or (len(thinking) + len(content)) // 4
                log(
                    f"  ⛔ {role} hit the context wall "
                    f"(num_ctx={options['num_ctx']}, generated {spent} tokens, "
                    f"~{len(thinking) // 4} of them thinking).",
                    verbose,
                )
                # A pi-style compaction rescue, when there is something to
                # resume from: the trace gets summarised, the answer-so-far
                # is kept verbatim, and the model is asked to finish from the
                # cut — as many passes as it takes, each truncated
                # continuation feeding the next (see _resume_compacted).
                # Any failure falls through to "ceiling" below, which the
                # callers already handle.
                if COMPACT_ENABLED and (thinking.strip() or content.strip()):
                    resumed = _resume_compacted(
                        role, system_prompt, user_prompt, thinking, content,
                        want_think, verbose,
                    )
                    if resumed:
                        log(f"  ✅ {role} completed after compaction + "
                            f"continuation.", verbose)
                        return resumed, ""
                return "", "ceiling"

            log(f"  ⚠️  {role} returned empty content (attempt {attempt}).", verbose)
        except (requests.RequestException, ValueError, KeyError) as e:
            log(f"  ⚠️  {role} transport error (attempt {attempt}): {e}", verbose)

    # Retries exhausted on empty replies or transport errors; nothing to salvage.
    return "", ""


def extract(
    instruction: str,
    source_text: str,
    schema: Dict[str, Any],
    role: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Stage 2: convert stage-1 prose into schema-conformant JSON.

    Thinking is never requested here, and the backend goes further and
    disables reasoning outright for this call, because on llama.cpp it is
    thinking that voids the grammar. That is the backend's problem, not
    this function's.
    """
    system = (
        "You convert a mathematician's written work into JSON. Copy the "
        "content faithfully: do not add claims, do not evaluate the "
        "mathematics, do not summarise beyond what is asked. "
        f"{instruction}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": source_text},
    ]

    # Schema-constrained first; drop to prompt-only if the grammar and the
    # model's default thinking mode conflict (symptom: empty content, output
    # stranded in message.thinking). The repair pipeline in clean_json_text
    # reclaims JSON from free-form output on the fallback path. The conflict
    # is a property of the model, not of one call, so once
    # discovered it's remembered for the rest of the run.
    global _SCHEMA_MODE_BROKEN
    use_schema = not _SCHEMA_MODE_BROKEN

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        options = dict(EXTRACT_OPTIONS)
        call_messages = messages
        if not use_schema:
            # Free-form mode: the schema moves from grammar to prompt.
            call_messages = [
                {
                    "role": "system",
                    "content": system
                    + " Respond with ONLY a JSON object matching this schema, "
                    "no fences, no commentary: "
                    + json.dumps(schema),
                },
            ] + messages[1:]
        # The extraction is a copy of the source, so it can be as long as the
        # source was; give it whatever this (larger) prompt leaves of the
        # window. That may be less than the source itself — the source plus a
        # copy of it do not always both fit — which is window physics, not a
        # budget to be nudged.
        options["num_predict"] = _headroom(call_messages, options["num_ctx"])

        content: Optional[str] = None
        try:
            # think is left as None throughout: whichever server we are on,
            # this stage wants the grammar honoured, and the backend knows
            # what that costs. No new mathematics happens here.
            reply = BACKEND.chat(
                call_messages,
                think=None,
                schema=schema if use_schema else None,
                options=options,
            )
            content = reply.content or ""
            if reply.truncated:
                log(f"  ⚠️  {role} extraction hit the token ceiling.", verbose)

            if use_schema and not content.strip():
                # Grammar/thinking conflict: qwen3.x thinks by default, the
                # format grammar suppresses the answer, and the output lands
                # in the thinking channel. Deterministic, so don't retry
                # same-mode.
                stranded = len(reply.thinking or "")
                log(
                    f"  ⚠️  {role} extraction returned empty content with format "
                    f"set ({stranded} chars stranded in thinking). Falling back "
                    f"to prompt-only JSON for the rest of the run.",
                    verbose,
                )
                use_schema = False
                _SCHEMA_MODE_BROKEN = True
                continue

            return json.loads(clean_json_text(content))
        except (requests.RequestException, KeyError, TypeError) as e:
            log(f"  ⚠️  {role} extraction transport error (attempt {attempt}): {e}", verbose)
        except json.JSONDecodeError as e:
            log(f"  ⚠️  {role} extraction parse error (attempt {attempt}): {e}", verbose)
            if content is not None:
                log(f"     raw ({len(content)} chars) head: {content[:200]!r}", verbose)
            if content is not None and content.strip():
                messages = messages[:2] + [
                    {"role": "assistant", "content": content[:1500]},
                    {
                        "role": "user",
                        "content": (
                            f"That was not valid JSON ({e}). Reply with ONLY the "
                            "JSON object, no fences and no commentary."
                        ),
                    },
                ]

    log(f"  ❌ {role} extraction failed after all retries.", verbose)
    return {}


# ----------------------------------------------------------------------------
# Context filtering
# ----------------------------------------------------------------------------
def planner_dag_view(dag: Dict[str, Any]) -> Dict[str, Any]:
    """Planner sees lemma statements + dependency structure, never proofs."""
    return {
        "proved_lemmas": {
            lid: {
                "statement": node["statement"],
                "dependencies": node.get("dependencies", []),
            }
            for lid, node in dag["lemmas"].items()
        }
    }


def prover_context(dag: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the prover is allowed to cite: every proved statement.

    This used to be the direct dependencies' full proofs plus a statement
    index, on the strength of the planner having declared which lemmas the
    argument would need. With that declaration gone there is no way to know in
    advance which proofs would be worth shipping, and shipping all of them
    grows without bound — prompt tokens and generation tokens share num_ctx,
    so by lemma thirty the prover would be paying to re-read every proof it
    has ever written in order to have room to write one more.

    The cost is real: the prover sees *that* a lemma holds, not how it was
    established. If a conjecture turns out to need the latter, the fix is a
    second prover call — one to ask which lemmas it wants, one to prove with
    those proofs attached — not to widen this.
    """
    return {
        "proved_lemmas": {
            lid: {"statement": node["statement"]}
            for lid, node in dag["lemmas"].items()
        }
    }


def verifier_context(
    dag: Dict[str, Any],
    cited: List[str],
    references: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Statements of exactly the results the prover claims to have used.

    DAG lemmas and reference results are merged into one set, keyed by id and
    carrying the statement only. Deliberately not the whole DAG or the whole
    reference collection: the verifier agents are asked to reject "use of
    results not
    present in the provided set", which only bites if the set is the prover's
    declared citations: a proof leaning on a result it never declared then
    reads as an unjustified leap, which is what it is.
    """
    statements = {
        lid: {"statement": dag["lemmas"][lid]["statement"]}
        for lid in cited
        if lid in dag["lemmas"]
    }
    for ref in references:
        lid = str(ref.get("id") or "")
        if lid in cited and lid not in statements:
            statements[lid] = {
                "statement": str(ref.get("formal statement", ""))
            }
    return {"cited_results": statements}


def scan_citations(
    proof: str, dag: Dict[str, Any], references: List[Dict[str, Any]]
) -> List[str]:
    """Fallback edge recovery: which known ids appear in the proof text.

    Only used when the prover ignored its output format entirely, in which
    case the alternative is a node with no edges at all — a lemma that
    silently claims to stand on its own. Scans lemma ids and reference ids;
    word-boundary matching, so lemma_1 does not match inside lemma_10.
    """
    ids = set(dag["lemmas"])
    ids.update(str(r.get("id")) for r in references if r.get("id"))
    return [
        lid for lid in ids
        if re.search(rf"\b{re.escape(lid)}\b", proof)
    ]


def load_references() -> List[Dict[str, Any]]:
    """Load references.md — parsing.py's output.

    A strict JSON array of { id, slogan, "formal statement", reference,
    tags } objects, written once by parsing.py; this module only reads it.
    A missing file returns [], which is exactly the pre-features run: every
    call site degrades to the old prompts. A corrupt file warns and returns
    [] too — the right fix is to re-run parsing.py, not to hand the loop a
    subset. Ids (ref_N) are assigned by parsing.py, never here, so a prover
    that cites one is citing a name the file vouches for.
    """
    if not REFERENCES_FILE or not os.path.exists(REFERENCES_FILE):
        return []
    try:
        with open(REFERENCES_FILE, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text.startswith("```"):
            text = _FENCE_RE.sub("", text).strip()
        data = json.loads(text)
    except (OSError, json.JSONDecodeError) as e:
        log(f"⚠️  {REFERENCES_FILE} is not valid JSON ({e}); ignoring it. "
            f"Re-run parsing.py if you expected references here.")
        return []
    if not isinstance(data, list):
        log(f"⚠️  {REFERENCES_FILE} is not a JSON array; ignoring it.")
        return []
    return [
        r for r in data
        if isinstance(r, dict) and str(r.get("id") or "").strip()
    ]


# ----------------------------------------------------------------------------
# Candidate handling
# ----------------------------------------------------------------------------
def planner_candidates(planner_res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise a planner reply into a list of well-formed candidate lemmas.

    Accepts the single-lemma shape as well, so a conjecture directory holding
    an old per-conjecture planner.md override keeps working: it simply gets a
    shortlist of length one, and every path below treats that as a shortlist
    that happens to be short.

    Candidates missing an id or a statement are dropped rather than passed on
    as half-lemmas, and duplicate ids are collapsed — models asked for five
    distinct routes will occasionally give the same lemma two numbers.
    """
    raw = planner_res.get("candidate_lemmas")
    if not isinstance(raw, list):
        single = planner_res.get("next_lemma")
        raw = [single] if isinstance(single, dict) else []

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        lemma_id = str(item.get("id") or "").strip()
        statement = str(item.get("statement") or "").strip()
        if not lemma_id or not statement or lemma_id in seen:
            continue
        seen.add(lemma_id)
        # Any dependencies field is dropped rather than honoured: a planner
        # that volunteers one (an old per-conjecture override, or a model
        # embellishing the schema) must not get to pre-empt the prover. A
        # missing or unrecognised aim defaults to "proof" for the same
        # reason: the safe reading of a candidate that predates aims is the
        # ordinary one.
        aim = str(item.get("aim") or "").strip().lower()
        if aim not in ("proof", "counterexample"):
            aim = "proof"
        out.append({"id": lemma_id, "statement": statement, "aim": aim})
        if len(out) >= PLANNER_CANDIDATES:
            break
    return out


def screen_candidates(
    dag: Dict[str, Any], candidates: List[Dict[str, Any]]
) -> List[Tuple[Dict[str, Any], List[str]]]:
    """Pair each candidate with the reasons it cannot be used as it stands.

    Only one check survives now that candidates carry no dependencies: an id
    already in the DAG. Proving it again gains nothing, and writing it again
    would overwrite a node that other lemmas may already cite. An empty
    problem list means the candidate is ready for the prover.
    """
    screened: List[Tuple[Dict[str, Any], List[str]]] = []
    for cand in candidates:
        problems: List[str] = []
        if cand["id"] in dag["lemmas"]:
            problems.append(f"{cand['id']} is already in the DAG")
        screened.append((cand, problems))
    return screened


def log_candidates(
    screened: List[Tuple[Dict[str, Any], List[str]]],
    plan_summary: str,
    verbose: bool,
) -> None:
    """Print the shortlist in automation too, not just when a human is asked.

    A run you walked away from should still leave a record of the four roads
    not taken; without it the log shows a decision and none of the choice.
    """
    log("🗺  Planner shortlist:", verbose)
    log(interaction.render(screened, plan_summary), verbose)


def select_lemma(
    usable: List[Dict[str, Any]],
    dag: Dict[str, Any],
    conjecture: str,
    plan_summary: str,
    failed_attempts: Dict[str, List[str]],
    verbose: bool,
) -> Dict[str, Any]:
    """Ask the selector agent to pick, from the candidates that survived
    screening, the one it judges most likely to come through the prover and
    the verifiers.

    The planner's best-first order is the fallback, not the default: if the
    selector fails, runs out of room, or names a lemma it was not offered, the
    first offered candidate is taken — which is what automation did before the
    selector existed, and which keeps --mode auto able to make progress even
    when this extra call goes wrong.
    """
    selector_sys = load_file(PROMPT_PATHS["selector.md"])
    selector_user = (
        f"Conjecture:\n{conjecture}\n\n"
        f"Planner's strategy summary:\n{plan_summary or '(none)'}\n\n"
        f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag), indent=2)}\n\n"
        f"Previously rejected attempts (routes the prover and the verifiers "
        f"have already found wanting):\n"
        f"{json.dumps(failed_attempts, indent=2)}\n\n"
        f"Candidate lemmas, in the order the planner offered them. Exactly "
        f"one will be sent to the prover; choose from these only:\n"
        f"{json.dumps(usable, indent=2)}"
    )
    text, _status = reason(selector_sys, selector_user, "selector",
                           THINK["selector"], verbose)
    check_mode_toggle(verbose)

    res: Optional[Dict[str, Any]] = None
    if text:
        # selector.md demands raw JSON output; extraction is the fallback.
        res = parse_json_or_none(text)
        if res is None:
            res = extract(
                "Extract the selector's decision. selected_id must be one of "
                "the offered candidate ids, copied verbatim.",
                text,
                SELECTOR_SCHEMA,
                "selector",
                verbose,
            )
    res = res or {}
    selected_id = str(res.get("selected_id") or "").strip()
    for cand in usable:
        if cand["id"] == selected_id:
            log(f"🧭 Selector picked {cand['id']}.", verbose)
            return cand

    if not text or not selected_id:
        log(
            f"⚠️  Selector gave no usable choice; taking the first candidate "
            f"instead ({usable[0]['id']}).",
            verbose,
        )
    else:
        log(
            f"⚠️  Selector chose {selected_id!r}, which was not among the "
            f"candidates offered; taking the first candidate instead "
            f"({usable[0]['id']}).",
            verbose,
        )
    return usable[0]


def check_mode_toggle(verbose: bool = True) -> None:
    """Apply a hotkey press, if one has landed since the last check.

    Called at several points in an iteration rather than only before the menu,
    because a press lands whenever the operator changes their mind and that is
    usually somewhere in the middle of a long prover, verifier or reviser call.
    MODE is
    still only *read* at the planning step, so checking early changes nothing
    functionally — it means the log acknowledges the key at the moment you
    press it instead of twenty minutes later, which is the difference between
    a toggle that feels responsive and one you press twice because the first
    press seemed not to register.

    The press is a toggle, so it works from either mode: it is the way back to
    automation from anywhere in the loop except the menu itself, where the
    listener is paused and 'a' does the same job.
    """
    global MODE
    if not HOTKEY.take():
        return
    MODE = "auto" if MODE == "human" else "human"
    log(
        "\n▶ Automation resumed; planning runs unattended from here."
        if MODE == "auto"
        else "\n⏸ Manual control engaged; you pick at the next planning step.",
        verbose,
    )


def _run_verifier(
    role: str,
    system_prompt: str,
    user_prompt: str,
    verbose: bool,
) -> Tuple[str, str]:
    """One atomic verification step: a single call to one verifier agent.

    Returns (decision, justification). A pass that hits the context wall
    without a verdict, or returns nothing parseable, is a reject — a check
    that cannot be completed can never count as an acceptance, so the proof
    goes to the reviser rather than into the DAG on the strength of the
    other two verifiers.
    """
    review, review_status = reason(
        system_prompt, user_prompt, role, THINK[role], verbose,
    )
    check_mode_toggle(verbose)
    if review_status == "ceiling" and not review:
        return (
            "reject",
            f"{role} exhausted its token budget without reaching a verdict; "
            "the proof is likely too long to review in one pass.",
        )
    # The verifier prompts demand raw JSON output; extraction is the fallback.
    res = parse_json_or_none(review) if review else None
    if res is None and review:
        res = extract(
            "Extract the verdict. decision is 'accept' only if the review "
            "endorses the proof without unresolved objections.",
            review,
            VERIFIER_SCHEMA,
            role,
            verbose,
        )
    res = res or {}
    decision = str(res.get("decision", "")).strip().lower()
    justification = str(res.get("justification") or "(no justification)").strip()
    return decision, justification


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run_loop(verbose: bool = True) -> Dict[str, Any]:
    """Execute the multi-agent theorem proving loop.

    Args:
        verbose: If True, print progress to the console.

    Returns:
        The final DAG, so callers running with --no-verbose still get a result.
    """
    global MODE            # the hotkey and the menu both retarget it mid-run

    conjecture = load_file(CONJECTURE_FILE)
    if not conjecture:
        log(f"{CONJECTURE_FILE} is empty — write your conjecture there first.", verbose)
        return load_dag()

    planner_sys = load_file(PROMPT_PATHS["planner.md"])
    selector_sys = load_file(PROMPT_PATHS["selector.md"])
    prover_sys = load_file(PROMPT_PATHS["prover.md"])
    # One prompt per verification step; VERIFIER_AGENTS names the three.
    verifier_sys = {name: load_file(PROMPT_PATHS[name]) for name in VERIFIER_AGENTS}
    reviser_sys = load_file(PROMPT_PATHS["reviser.md"])

    empty = [name for name, text in (
        ("planner.md", planner_sys), ("selector.md", selector_sys),
        ("prover.md", prover_sys), *verifier_sys.items(),
        ("reviser.md", reviser_sys),
    ) if not text]
    if empty:
        # A missing system prompt does not crash — it produces an agent with
        # no instructions, which fails in ways that look like model problems.
        log(f"Missing or empty prompt files: {', '.join(empty)}. Aborting.", verbose)
        return load_dag()

    # references.md, if parsing.py produced one. The prover gets the full
    # collection unconditionally — it is the tool that cites — so it is
    # rendered once here and reused on every attempt.
    references = load_references()
    ref_ids = {str(r["id"]) for r in references}
    prover_refs = [
        {
            "id": r["id"],
            "slogan": r.get("slogan", ""),
            "formal statement": r.get("formal statement", ""),
            "tags": r.get("tags", []),
        }
        for r in references
    ]
    reference_block = (
        "\n\nKnown references (theorem-level results from the parsed "
        "collection; you may cite any of them by id in cited_lemmas without "
        "proving them yourself):\n"
        + json.dumps(prover_refs, indent=2)
        if prover_refs
        else ""
    )
    # The planner only sees a small collection: below PLANNER_REFERENCE_LIMIT
    # it gets an id + slogan per result, at or above it none. See the limit's
    # comment for why a blunt rule beats a graded one here.
    if references:
        if len(references) < PLANNER_REFERENCE_LIMIT:
            planner_ref_block = (
                "\n\nKnown references (named theorem-level results the prover "
                "may cite by id; the prover sees their full statements):\n"
                + json.dumps(
                    [
                        {
                            "id": r["id"],
                            "slogan": r.get("slogan", ""),
                            "tags": r.get("tags", []),
                        }
                        for r in references
                    ],
                    indent=2,
                )
            )
        else:
            planner_ref_block = ""
            log(
                f"ℹ️  {len(references)} references parsed; that is at or "
                f"above the planner limit ({PLANNER_REFERENCE_LIMIT}), so the "
                f"planner gets none. The prover sees them all."
            )
    else:
        planner_ref_block = ""

    failed_attempts: Dict[str, List[str]] = {}
    warned_dangling = set()

    for iteration in range(1, MAX_ITERATIONS + 1):
        log(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---", verbose)
        check_mode_toggle(verbose)
        dag = load_dag()

        # An existing DAG may cite reference ids; one that no longer exists in
        # references.md (a re-parse dropped it) dangles every proof that used
        # it, and the verifier would reject those for good reason. Warn once
        # per (lemma, dependency) pair rather than every iteration.
        known_ids = set(dag["lemmas"]) | ref_ids
        for lid, node in dag["lemmas"].items():
            for dep in node.get("dependencies", []):
                if dep not in known_ids and (lid, dep) not in warned_dangling:
                    warned_dangling.add((lid, dep))
                    log(
                        f"⚠️  DAG node {lid} cites {dep}, which is neither a "
                        f"lemma nor a parsed reference; verification of it "
                        f"will see a citation with no statement behind it."
                    )

        # ---------------- Step 1: Planner ----------------
        planner_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag), indent=2)}"
            f"{planner_ref_block}\n\n"
            f"Previously rejected attempts (avoid or decompose these):\n"
            f"{json.dumps(failed_attempts, indent=2)}"
        )
        plan_text, plan_status = reason(
            planner_sys, planner_user, "planner", THINK["planner"], verbose
        )
        check_mode_toggle(verbose)
        if plan_status == "ceiling" and not plan_text:
            log("Planner exhausted its token budget. Re-planning.", verbose)
            continue
        if not plan_text:
            log("Planner produced nothing. Re-planning next iteration.", verbose)
            continue

        # planner.md demands raw JSON output; extraction is only the fallback.
        planner_res = parse_json_or_none(plan_text)
        if planner_res is None:
            planner_res = extract(
                "Extract the plan. Copy every candidate lemma the text proposes "
                "into candidate_lemmas, preserving the order it presents them "
                "in and each candidate's aim ('proof' or 'counterexample'); "
                "where the text gives no aim, use 'proof'. "
                "is_conjecture_proved must be true only if the text "
                "explicitly concludes the conjecture is fully proved; "
                "is_conjecture_disproved must be true only if the text "
                "explicitly concludes that a full counterexample to the "
                "conjecture has been established.",
                plan_text,
                PLANNER_SCHEMA,
                "planner",
                verbose,
            )

        if planner_res.get("is_conjecture_proved"):
            log("\n🎉 Conjecture has been fully proved!", verbose)
            dag["conjecture"] = conjecture
            dag["status"] = "proved"
            save_dag(dag)
            return dag

        if planner_res.get("is_conjecture_disproved"):
            log(
                "\n💥 Conjecture has been disproved: a counterexample to it "
                "stands in the DAG.",
                verbose,
            )
            dag["conjecture"] = conjecture
            dag["status"] = "disproved"
            save_dag(dag)
            return dag

        candidates = planner_candidates(planner_res)
        if not candidates:
            log("Planner proposed no usable lemma. Re-planning.", verbose)
            continue

        screened = screen_candidates(dag, candidates)
        summary = str(planner_res.get("plan_summary") or "")

        # ---------------- Step 1b: Selection ----------------
        # The one place the two modes differ, and so the one place MODE is
        # read. Everything after this block runs identically whether the lemma
        # was chosen by a person or by the selector, which is what keeps
        # --mode auto honest as a control.
        next_lemma: Optional[Dict[str, Any]] = None
        if MODE == "human":
            choice = interaction.choose(
                screened, dag["lemmas"].keys(), HOTKEY, summary
            )
            if choice.action == "quit":
                log("\n⏹ Stopped by the operator; DAG kept as it stands.", verbose)
                save_dag(dag)
                return dag
            if choice.action == "replan":
                # Recorded against every candidate shown, because the channel
                # the planner reads is keyed by lemma id and the point is that
                # it should not come back with this same shortlist.
                for cand, _ in screened:
                    for note in choice.notes:
                        failed_attempts.setdefault(cand["id"], []).append(note)
                log("↩︎ Shortlist rejected; asking the planner again.", verbose)
                continue
            if choice.action == "auto":
                MODE = "auto"
                log(f"▶ Automation resumed. {HOTKEY.hint()}", verbose)
            elif choice.action == "assert":
                # The operator has vouched for it, so there is nothing for the
                # prover or the verifiers to do: no proof is generated, no
                # review is run, and the lemma is in the DAG before the next
                # iteration reloads it. `provenance` marks it as resting on a
                # person rather than on a machine-checked argument — absent on
                # every node written by the pipeline, and on every DAG file
                # that predates this, so read it with .get().
                asserted = choice.lemma or {}
                dag["lemmas"][asserted["id"]] = {
                    "statement": asserted["statement"],
                    "proof": "Asserted by the operator; not machine-proved.",
                    "dependencies": [],   # nothing was cited; nothing was proved
                    "provenance": "operator",
                }
                save_dag(dag)
                failed_attempts.pop(asserted["id"], None)
                log(
                    f"🖊  Lemma {asserted['id']} accepted on your authority "
                    f"and added to the DAG unproved.",
                    verbose,
                )
                continue
            else:
                next_lemma = choice.lemma

        if next_lemma is None:
            # Automatic selection. Screening removes the candidates whose ids
            # are already in the DAG; of what survives, the selector agent
            # weighs each one and names the one it judges most likely to
            # succeed.
            # The planner's best-first order is only the fallback: with
            # exactly one candidate surviving there is nothing to choose, and
            # select_lemma() itself falls back when the selector fails.
            log_candidates(screened, summary, verbose)
            usable = [cand for cand, problems in screened if not problems]
            if not usable:
                log("No candidate is currently provable; re-planning.", verbose)
                for cand, problems in screened:
                    failed_attempts.setdefault(cand["id"], []).append(
                        f"Planning error: {'; '.join(problems)}."
                    )
                continue
            if len(usable) == 1:
                next_lemma = usable[0]
                if next_lemma is not screened[0][0]:
                    log(
                        f"↷ Skipped {screened[0][0]['id']} "
                        f"({'; '.join(screened[0][1])}); took {next_lemma['id']}.",
                        verbose,
                    )
            else:
                next_lemma = select_lemma(
                    usable, dag, conjecture, summary, failed_attempts, verbose
                )

        lemma_id = next_lemma["id"]
        lemma_stmt = next_lemma["statement"]
        aim = str(next_lemma.get("aim") or "proof").strip().lower()
        if aim not in ("proof", "counterexample"):
            aim = "proof"

        log(f"📌 Next Lemma [{lemma_id}] (aim: {aim}): {lemma_stmt}", verbose)

        # ---------------- Steps 2-5: the proof loop ----------------
        # prover -> verifiers -> reviser, repeated within the iteration. The
        # prover writes a proof of `target`; the verifiers then check it —
        # three atomic passes, one per verifier agent — and one reject ends the
        # counting and sends the proof to the reviser with the verdict's
        # reasoning. The reviser either keeps the statement (the prover
        # re-tries with the verdict as feedback) or revises it (the revised
        # statement becomes the new target). Only a proof that survives all
        # three verifiers enters the DAG; after MAX_PROOF_ATTEMPTS prover
        # rounds the lemma is given up for this iteration and the planner is
        # asked again.
        #
        # No human intervention here, in either mode. Choosing 'p' at the menu
        # *is* the decision to let the models settle it; asking again
        # afterwards would put the operator back in the loop they just
        # delegated out of. If you want a say over this lemma, assert it.
        target: Dict[str, Any] = {"id": lemma_id, "statement": lemma_stmt}
        if aim != "proof":
            target["aim"] = aim
        # The prover sees only the most recent failure: feedback is rebuilt
        # after each rejected round, so a retry reads the last verdict and
        # diagnosis, not the history of every earlier attempt. That history
        # is kept for the planner in failed_attempts — decomposing or
        # rerouting is its job, not the prover's.
        feedback: List[str] = []
        attempt_notes: List[str] = list(failed_attempts.get(lemma_id, []))
        proved = False

        for attempt in range(1, MAX_PROOF_ATTEMPTS + 1):
            log(
                f"📝 Proof attempt {attempt}/{MAX_PROOF_ATTEMPTS} for {lemma_id}.",
                verbose,
            )

            # ---------------- Step 2: Prover ----------------
            # prover.md instructs the model to reply with {"lemma_id",
            # "cited_lemmas", "proof"}. We parse that JSON locally rather than
            # via a second model call, so the proof text can never be abridged
            # or paraphrased; if the model ignored the format, its content is
            # taken verbatim as the proof.
            prover_user = (
                f"Conjecture:\n{conjecture}\n\n"
                f"Available proved lemmas:\n"
                f"{json.dumps(prover_context(dag), indent=2)}"
                f"{reference_block}\n\n"
                f"Lemma to prove:\n{json.dumps(target, indent=2)}\n"
                + (
                    f"\nFeedback from the previous attempt "
                    f"(the most recent rejection only):\n"
                    f"{json.dumps(feedback, indent=2)}"
                    if feedback
                    else ""
                )
            )
            proof_text, prover_status = reason(
                prover_sys, prover_user, "prover", THINK["prover"], verbose
            )
            check_mode_toggle(verbose)

            if prover_status == "ceiling" and not proof_text:
                log(
                    f"⛔ Lemma {lemma_id} is too large to prove in one call. "
                    f"Asking the planner to decompose it.",
                    verbose,
                )
                attempt_notes.append(
                    "Lemma too large: the prover exhausted its entire token "
                    "budget without completing a proof. Decompose this into "
                    "smaller, independently provable lemmas rather than "
                    "re-proposing it."
                )
                break

            proof = proof_text
            declared: Optional[List[str]] = None
            prover_res = parse_json_or_none(proof_text)
            if prover_res is not None:
                candidate = prover_res.get("proof")
                if isinstance(candidate, str) and candidate.strip():
                    proof = candidate.strip()
                raw_cited = prover_res.get("cited_lemmas")
                if isinstance(raw_cited, list):
                    declared = [str(c).strip() for c in raw_cited if str(c).strip()]

            if not proof:
                log(f"❌ Prover produced no proof for {lemma_id}.", verbose)
                attempt_notes.append("Prover returned nothing.")
                break

            # The DAG's edges now come from here. An empty declared list is
            # taken at face value — a proof from first principles has no
            # dependencies — but a missing one means the prover ignored its
            # format, and scanning the text beats recording a lemma as
            # standing on nothing.
            if declared is None:
                dep_ids = scan_citations(proof, dag, references)
                if dep_ids:
                    log(
                        f"  ℹ️  Prover declared no citations; recovered "
                        f"{', '.join(dep_ids)} from the proof text.",
                        verbose,
                    )
            else:
                # Citations may name DAG lemmas or parsed references; the
                # verifier's context is the union of the two, so both are
                # kept. A name in neither set is dropped, and the verifier is
                # about to see a proof that leans on a result absent from its
                # context, which is exactly the unjustified step it is meant
                # to catch.
                dep_ids = [c for c in declared if c in dag["lemmas"] or c in ref_ids]
                phantom = [
                    c for c in declared if c not in dag["lemmas"] and c not in ref_ids
                ]
                if phantom:
                    log(
                        f"  ⚠️  Prover cited ids that are neither lemmas nor "
                        f"references: {', '.join(phantom)}.",
                        verbose,
                    )

            log(
                f"✍️ Proof generated for {lemma_id} ({len(proof)} chars"
                f"{', cites ' + ', '.join(dep_ids) if dep_ids else ', no citations'}).",
                verbose,
            )

            # ---------------- Step 3: the three verifier checks -----------
            # Three atomic checks, one per verifier agent, in the order
            # verifier_1, verifier_2, verifier_3. Each is a single call to its
            # own agent; one reject ends the counting and carries its
            # reasoning to the reviser; only a proof that survives all three
            # is accepted.
            verifier_user = (
                f"Conjecture:\n{conjecture}\n\n"
                f"Cited results (statements of exactly the lemmas and "
                f"references the proof declares it used; nothing else is "
                f"available to it):\n"
                f"{json.dumps(verifier_context(dag, dep_ids, references), indent=2)}\n\n"
                f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
                f"Proposed proof:\n{proof}"
            )
            reject_just: Optional[str] = None
            for step, agent in enumerate(VERIFIER_AGENTS, 1):
                role = agent[:-3]   # "verifier_1.md" -> "verifier_1"
                decision, justification = _run_verifier(
                    role, verifier_sys[agent], verifier_user, verbose,
                )
                log(
                    f"🔍 Verifier {step}/{len(VERIFIER_AGENTS)} ({role}): "
                    f"{decision.upper() or '???'} — {justification}",
                    verbose,
                )
                if decision != "accept":
                    reject_just = justification
                    break

            if reject_just is None:
                # ---------------- Step 4: DAG update ----------------
                # All three verifier checks accepted the same proof.
                log(
                    f"✅ Lemma {lemma_id} passed all {len(VERIFIER_AGENTS)} "
                    f"verifier checks. Adding to DAG.",
                    verbose,
                )
                dag["lemmas"][lemma_id] = {
                    "statement": target["statement"],
                    "proof": proof,
                    "dependencies": dep_ids,
                }
                save_dag(dag)
                failed_attempts.pop(lemma_id, None)
                proved = True
                break

            # ---------------- Step 5: Reviser ----------------
            log(
                f"❌ Proof rejected for {lemma_id}. Sending the proof and the "
                f"verdict to the reviser.",
                verbose,
            )
            reviser_user = (
                f"Conjecture:\n{conjecture}\n\n"
                f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
                f"Rejected proof:\n{proof}\n\n"
                f"Verifier's reasoning (why the proof failed):\n{reject_just}"
            )
            revision_text, _revision_status = reason(
                reviser_sys, reviser_user, "reviser", THINK["reviser"], verbose
            )
            check_mode_toggle(verbose)

            revision_res: Optional[Dict[str, Any]] = None
            if revision_text:
                # reviser.md demands raw JSON output; extraction is the
                # fallback.
                revision_res = parse_json_or_none(revision_text)
                if revision_res is None:
                    revision_res = extract(
                        "Extract the revision decision. statement_revision is "
                        "true only if the text proposes a changed statement. "
                        "Copy new_statement verbatim; it is the empty string "
                        "when the statement is kept.",
                        revision_text,
                        REVISER_SCHEMA,
                        "reviser",
                        verbose,
                    )
            else:
                log(
                    "⚠️  Reviser returned nothing; the verdict alone goes back "
                    "to the prover.",
                    verbose,
                )
            revision_res = revision_res or {}

            diagnosis = str(revision_res.get("diagnosis") or "").strip()
            new_stmt = str(revision_res.get("new_statement") or "").strip()
            wants_revision = bool(revision_res.get("statement_revision", False))

            feedback = [f"Verifier: {reject_just}"]
            if diagnosis:
                feedback.append(f"Reviser's diagnosis: {diagnosis}")

            if wants_revision and new_stmt and new_stmt != target["statement"]:
                target = {"id": lemma_id, "statement": new_stmt}
                if aim != "proof":
                    target["aim"] = aim
                log(
                    f"↻ Reviser revised the lemma; the prover starts again "
                    f"from:\n{new_stmt}",
                    verbose,
                )
                feedback.append(f"Revised statement proposed: {new_stmt}")
            elif wants_revision:
                log(
                    "⚠️  Reviser flagged a statement revision but gave none "
                    "usable; keeping the statement.",
                    verbose,
                )
            else:
                log(
                    "↻ Reviser kept the statement; the prover re-tries with "
                    "the verdict as feedback.",
                    verbose,
                )
            attempt_notes.extend(feedback)
            # Loop continues: the next prover round works on `target`, with
            # only this round's feedback.

        # ---------------- Step 6: Record the outcome for the planner --------
        if not proved:
            if attempt_notes:
                failed_attempts[lemma_id] = attempt_notes
            log(
                f"↷ Lemma {lemma_id} left unproved after the proof loop; the "
                f"planner will see the feedback.",
                verbose,
            )

    log(
        f"\n⏹ Reached MAX_ITERATIONS ({MAX_ITERATIONS}) without settling the "
        f"conjecture (proved or disproved).",
        verbose,
    )
    return load_dag()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    global MODEL_NAME, CONJECTURE_FILE, DAG_FILE, REFERENCES_FILE
    global MAX_ITERATIONS, NUM_CTX
    global BACKEND, PROFILE, PROMPT_PATHS, MODE, HOTKEY
    global MAX_PROOF_ATTEMPTS

    parser = argparse.ArgumentParser(
        description="Run the multi-agent theorem prover."
    )
    # Adds --verbose / --no-verbose flags (defaults to True)
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable console logging (default: --verbose)",
    )
    parser.add_argument(
        "--host", default=None,
        help="llama-server base URL. Defaults to $LLAMA_HOST or "
             "localhost:8081 (8080 is taken by open-webui).",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help="The --alias llama-server was launched with",
    )
    parser.add_argument(
        "--conjecture", default=DEFAULT_CONJECTURE,
        help=f"Directory under {workspace.CONJECTURES_ROOT}/ containing "
             f"{workspace.CONJECTURE_FILENAME} (default: {DEFAULT_CONJECTURE})",
    )
    parser.add_argument(
        "--dag", default=None,
        help="Override the DAG path. By default every run of a conjecture "
             "shares conjectures/<name>/dag.json, whatever model wrote it.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS,
        help=f"Loop iterations before giving up (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--max-proof-attempts", type=int, default=MAX_PROOF_ATTEMPTS,
        help=(
            "How many prover rounds the proof loop may spend on one lemma "
            f"per iteration (default: {MAX_PROOF_ATTEMPTS})"
        ),
    )
    parser.add_argument(
        "--num-ctx", type=int, default=None,
        help=(
            "Context window in tokens. Defaults to the server's own — "
            "the -c llama-server was launched with, as probed. Lower it if "
            "VRAM is tight."
        ),
    )
    parser.add_argument(
        "--mode", choices=("auto", "human"), default=DEFAULT_MODE,
        help="auto: the loop picks a lemma from the planner's shortlist and "
             "runs unattended (default). human: the shortlist is put to you at "
             "every planning step. Switchable mid-run either way.",
    )
    parser.add_argument(
        "--hotkey", default=HOTKEY_KEYS,
        help=f"Key that toggles between manual and automatic mid-run "
             f"(default: {HOTKEY_KEYS!r}). Pass '' to disable the listener.",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model
    MAX_ITERATIONS = args.max_iterations
    MAX_PROOF_ATTEMPTS = max(1, args.max_proof_attempts)
    MODE = args.mode

    # Backend first: the probe tells us the real context ceiling, which an
    # explicit --num-ctx is clamped to. Doing this before resolving paths
    # also means a dead server is reported before a missing directory.
    BACKEND = llm_backend.make_backend(
        args.model, args.host, REQUEST_TIMEOUT
    )
    PROFILE = BACKEND.probe()

    # Resolve the context window: an explicit --num-ctx wins (clamped to the
    # probed ceiling); with no argument, take the server's own context — see
    # the NUM_CTX comment.
    if args.num_ctx is None:
        NUM_CTX = PROFILE.context_limit
    else:
        NUM_CTX = min(args.num_ctx, PROFILE.context_limit)
    REASONING_OPTIONS["num_ctx"] = NUM_CTX
    EXTRACT_OPTIONS["num_ctx"] = NUM_CTX
    # The banner warns when an *explicit* --num-ctx is clamped; in the auto
    # case the resolution above has already made the two agree.
    log(llm_backend.describe(
        PROFILE, args.num_ctx if args.num_ctx is not None else NUM_CTX
    ), args.verbose)

    try:
        paths = workspace.resolve(args.conjecture, args.dag)
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
        return  # unreachable; parser.error exits, but keeps type checkers calm

    CONJECTURE_FILE = str(paths.conjecture)
    DAG_FILE = str(paths.dag)
    REFERENCES_FILE = str(paths.references)
    PROMPT_PATHS = {name: str(p) for name, p in paths.prompts.items()}
    log(workspace.describe(paths), args.verbose)

    # The listener puts the terminal in cbreak mode, so it has to be stopped on
    # every exit path — including a traceback — or the shell you return to has
    # no echo. Hence try/finally rather than a stop() at the end of the run.
    HOTKEY = interaction.HotKey(
        args.hotkey,
        enabled=bool(args.hotkey),
        on_press=lambda: print(
            f"\n⇄ Mode change queued: "
            f"{'automatic' if MODE == 'human' else 'manual'} "
            f"from the next planning step."
        ),
    ).start()
    if MODE == "auto" and HOTKEY.hint():
        log(f"mode=auto {HOTKEY.hint()}", args.verbose)
    elif MODE == "human":
        log("mode=human — you pick the lemma at every planning step.", args.verbose)

    try:
        run_loop(verbose=args.verbose)
    finally:
        HOTKEY.stop()


if __name__ == "__main__":
    main()
