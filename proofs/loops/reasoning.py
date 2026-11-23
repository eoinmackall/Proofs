"""Model calls: free-form reasoning (reason), extraction into a schema
(extract), and the context-wall rescue between them.
"""

from __future__ import annotations

import contextlib
import email.utils
import json
import random
import re
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import requests

from .. import llm
from .. import tools as tools_mod
from ..workspace import settings as settings_mod
from .json_repair import _loads, clean_json_text, strip_thinking

REQUEST_TIMEOUT = 5400       # 90 minutes: thinking models are slow, but a


class Cancelled(Exception):
    """The caller stopped waiting for this call (a verifier whose verdict
    no longer matters: another one rejected the proof first). Raised in
    place of the next request, never in the middle of one."""


def _chat(ctx, m: "llm.Model", *args: Any, **kwargs: Any) -> "llm.Reply":
    """m's backend.chat, unless the calling thread has been cancelled
    (LoopState.cancel): every request of reason() and extract() goes
    through here, so a cancelled verifier sends no further tool round,
    retry or compaction pass. A context-length refusal is tagged with m,
    whose window it was (_fail_context_length)."""
    cancel = ctx.loop.cancel
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    try:
        return m.backend.chat(*args, **kwargs)
    except (llm.ContextLengthError, llm.QuotaExhausted) as e:
        e.model = m
        raise


def _retry_after(e: Exception) -> Optional[float]:
    """The seconds an HTTP error's Retry-After header asks for, or None."""
    response = getattr(e, "response", None)
    value = response.headers.get("Retry-After") if response is not None else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def _retry_pause(ctx, attempt: int, e: Exception, label: str,
                 verbose: bool) -> None:
    """Wait before retry number attempt + 1 of a request that failed in
    transport (see RETRY_BASE_SECONDS), saying so first. A cancelled
    verifier (LoopState.cancel) stops waiting at once: its next request
    raises Cancelled."""
    delay = min(RETRY_CAP_SECONDS, RETRY_BASE_SECONDS * 2 ** (attempt - 1))
    # Jitter: verifiers refused together do not retry together.
    delay *= random.uniform(0.75, 1.25)
    asked = _retry_after(e)
    if asked is not None:
        delay = max(delay, min(asked, RETRY_AFTER_CAP_SECONDS))
    ctx.ui.log(f"  {label}: retrying in {delay:.0f}s.", verbose)
    cancel = ctx.loop.cancel
    if cancel is not None:
        cancel.wait(delay)
    else:
        time.sleep(delay)


def _native_tools(ctx, m: "llm.Model") -> bool:
    return ctx.settings.tool_mode == "auto" and m.native_tools_ok is not False


def _disable_native_tools(ctx, m: "llm.Model", why: str, verbose: bool) -> None:
    if m.native_tools_ok is not False:
        m.native_tools_ok = False
        ctx.ui.log(
            f"The server of {m.label()} refused native tool calls "
            f"({why[:200]}); using the JSON tool protocol with it for the "
            f"rest of the run. Launch llama-server with --jinja for native "
            f"tool calls.",
            verbose,
        )


# How long the proof loop waits before retrying a call the server never
# answered (reason()'s "unavailable"): an outage is not the model's
# failure, so the round is not counted, and the call is retried until the
# server is back — or the run is stopped (Ctrl-C, or a parallel stop),
# with the round kept in the checkpoint.
SERVER_WAIT_SECONDS = 60

# The pause before retrying a request that failed in transport (a refused
# connection, a 429 or a 503, a reply that was not JSON): RETRY_BASE_SECONDS,
# doubling with each attempt up to RETRY_CAP_SECONDS, or longer when the
# server's Retry-After asks for it, up to RETRY_AFTER_CAP_SECONDS. Retrying
# at once only spends the attempts on a server that is still busy, and the
# parallel verifiers would all come back at the same moment.
RETRY_BASE_SECONDS = 2
RETRY_CAP_SECONDS = 60
RETRY_AFTER_CAP_SECONDS = 300


# Every role reason() and extract() are called with: the roles llm.routes
# routes (llm/routing.py). There is no fallback, so a new agent is listed
# here. And the ones that must share a model: every verifier, since a
# certificate records one model for the whole check. VERIFIER_ROLE stands
# for them all where a verifier's model is recorded or compared.
ROLES = (
    "planner", "prover", "verifier_1", "verifier_2", "verifier_3",
    "coding_verifier", "reviser", "parser", "parallel_planner",
    "parallel_lemma_generator",
)
VERIFIER_ROLES = ("verifier_1", "verifier_2", "verifier_3", "coding_verifier")
VERIFIER_ROLE = VERIFIER_ROLES[0]
SAME_MODEL = (
    (VERIFIER_ROLES, "a certificate records one model for the whole check"),
)


# extract()'s sampling: greedy, with every penalty off. The stage copies its
# source — it writes nothing new — and a server's launch-time sampling (a
# model card's presence penalty, say) would push a copy of mathematics,
# which repeats its symbols, to paraphrase. The one exception to leaving
# sampling to the server; sent only where the API takes it, and dropped
# for the rest of the run by a server that refuses it.
_GREEDY_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
    "min_p": 0.0,
}


def _call_options(m: "llm.Model") -> Dict[str, Any]:
    """The options of every call on m (reason(), extract(), a compaction
    pass). Sampling is left to the server and the model's own card: no
    temperature, penalty or sampler field is sent unless the model's entry
    sets one (llm.models.NAME.temperature, or --temperature) — a model's
    defaults are tuned for it, and a strict API (current Claude models,
    OpenAI's reasoning models) refuses a field it does not take.
    Likewise the reasoning effort (llm.models.NAME.effort, or --effort),
    which each backend sends in its API's own field. num_predict is set per
    call."""
    options: Dict[str, Any] = {"ctx_budget": m.ctx_budget}
    if m.spec.temperature is not None:
        options["temperature"] = m.spec.temperature
    if m.spec.effort is not None:
        options["effort"] = m.spec.effort
    return options


# Pi-style compaction on the context wall: a truncated call is resumed by
# summarising its thinking trace and re-sending task + answer-so-far (see
# _resume_compacted, between _headroom and reason below). A truncated
# continuation is not a failure but the next pass; the passes run until the
# answer completes, nothing is left to build on, or llm.max_compaction_passes is
# spent. COMPACT_SUMMARY_TOKENS is the room a compaction call keeps for the
# summary it writes; the trace fills the rest of the window (_compact_budget),
# so a trace is split into chunks only when it cannot fit one call.
# COMPACT_TAIL_CHARS is what the compactor sees of the answer-so-far,
# COMPACT_MIN_ROOM is the smallest headroom a continuation is worth attempting.
#
# COMPACT_THRESHOLD is the fraction of the window a call is allowed to fill
# (prompt + thinking + answer) before it is cut off and compacted. It is the
# single knob for "how long a role may think before compaction": the generation
# budget is set so that prompt + budget lands on COMPACT_THRESHOLD * ctx_budget,
# leaving (1 - COMPACT_THRESHOLD) of the window as a safety margin rather than
# a fixed 512 tokens. 0.99 leaves a ~1% margin; note the prompt itself eats
# into the window, so the generation budget is really
# (COMPACT_THRESHOLD * ctx_budget) - prompt — a 30k prompt on a 90k window caps
# generation near 60k regardless of this value. Raise toward 1.0 for maximum
# thinking room; lower it only if a run reports the server tripping the wall.
COMPACT_ENABLED = True


COMPACT_THRESHOLD = 0.99


COMPACT_SUMMARY_TOKENS = 16384


COMPACT_TAIL_CHARS = 2000


COMPACT_MIN_ROOM = 2048


# Room kept for the reply. ctx_budget is enforced only through the max_tokens
# reason() asks for — the server never sees it — so nothing but this keeps a
# prompt that grows with tool results from leaving a thinking model too
# little room to think and then answer. RESERVE_TOKENS (Pi's default
# reserveTokens) is the room no lookup may use: once the prompt reaches it,
# reason() closes the tools and the model answers, and a result that would
# cut into it is trimmed to fit (tools.ToolSet.run). LOOKUP_ROOM is room for
# a round or two of lookups on top of that: _fit_prompt shrinks the lemma
# window until a prompt leaves both. Each is clamped to a quarter of a small
# window, so the prompt keeps at least half of it (_reserve, _lookup_room).
RESERVE_TOKENS = 16384


LOOKUP_ROOM = 16384


# What a tool result's framing costs beyond its own text: the tool message
# around it, the room line it ends with, the JSON protocol's instruction.
_RESULT_FRAME_TOKENS = 128


# The least room before the reserve worth offering another lookup for —
# about one lemma statement. Below it, a call could only come back as
# tools.NO_ROOM, so the lookups close a little before the reserve itself.
_MIN_RESULT_TOKENS = 256


def _reserve(ctx_budget: int) -> int:
    return min(RESERVE_TOKENS, ctx_budget // 4)


def _lookup_room(ctx_budget: int) -> int:
    return min(LOOKUP_ROOM, ctx_budget // 4)


# HTTP statuses that mean "this request is malformed", not "the server is
# down": answered by changing the request, never by waiting.
_REJECTED_STATUSES = (400, 422)


def _model_for(ctx, role: str) -> "llm.Model":
    """The model role's calls go to on this thread: the model it is pinned
    to (a failed-over call, a verification check restarted on the
    fallback: LoopState.pinned), else the one it is routed to."""
    return ctx.loop.pinned.get(role) or ctx.models.for_role(role)


def _stage_model(ctx, stage: str, role: str) -> "llm.Model":
    """The model a stage of role's call goes to: the stage's own route,
    else role's model (_model_for)."""
    return ctx.models.stage_route(stage) or _model_for(ctx, role)


class VerifierFailover(Exception):
    """A verifier's call failed over (--with-fallback). A certificate names
    one model for its whole check, so the call is not sent to the fallback
    on its own: the check catches this, pins every verifier to model and
    runs again from the start (verifiers.with_verifier_failover)."""

    def __init__(self, model: "llm.Model", why: str) -> None:
        super().__init__(why)
        self.model = model
        self.why = why


def _fallback_for(ctx, m: "llm.Model") -> Optional["llm.Model"]:
    """The model a call that m could not serve goes to: the fallback
    (--with-fallback), unless there is none or m is it — then there is
    nothing to fall back to, and the call keeps its retry behaviour."""
    fallback = ctx.models.fallback if ctx.models is not None else None
    return None if fallback is None or fallback is m else fallback


@contextlib.contextmanager
def pinned(ctx, pins: Mapping[str, "llm.Model"]) -> Iterator[None]:
    """ctx's thread with pins added to its pinned roles, put back as they
    were on the way out."""
    saved = dict(ctx.loop.pinned)
    ctx.loop.pinned.update(pins)
    try:
        yield
    finally:
        ctx.loop.pinned.clear()
        ctx.loop.pinned.update(saved)


def _fail_quota(ctx, e: "llm.QuotaExhausted", stop: Any = None) -> None:
    """Run-level handling of a spent quota with nothing to fall back to:
    one message per model, then the stop the caller names. Logged
    unconditionally, like a context-length stop."""
    m = e.model
    said = f"quota:{m.name}" if m is not None else "quota"
    if said not in ctx.run.said:
        ctx.run.said.add(said)
        who = m.label() if m is not None else "the model"
        ctx.ui.log(
            f"\nThe API of {who} refused a call: its quota, credit or spend "
            f"limit is used up ({e}). Waiting does not bring it back: top it "
            f"up, or route its roles to another model. The checkpoint is "
            f"safe; re-running resumes from the last safe boundary."
        )
    if stop is not None:
        stop()


def _served_model(ctx, role: str) -> str:
    """The recorded name of the model that served role's latest answer on
    this thread (a failed-over call's fallback), else _recorded_model."""
    m = ctx.loop.served.get(role)
    return m.recorded() if m is not None else _recorded_model(ctx, role)


def _recorded_model(ctx, role: str) -> str:
    """The model name certificates and refutations record for work role
    did: the model routed to it, as its backend's recorded_model() reads it
    — the file a llama-server loaded, or the model a hosted API's reply
    names — rather than the request, which a llama-server ignores. "" for
    a command that calls no model."""
    return _model_for(ctx, role).recorded() if ctx.models is not None else ""


def _requested_model(ctx, role: str) -> str:
    """The model name role's requests carry, saved beside the recorded one
    (_same_model)."""
    return _model_for(ctx, role).requested if ctx.models is not None else ""


def _same_model(ctx, role: str, model: str, requested: str) -> bool:
    """Whether work recorded with (model, requested) — the recorded model
    and the request of the run that did it — came from the model this run
    routes role to. By the recorded model whenever the server has said
    what it runs (a llama-server ignores the request, so it names nothing
    there); by the request only before it has, when the recorded model is
    the request itself and a hosted API's reply would add a version suffix
    the saved name has."""
    m = _model_for(ctx, role)
    if m.recorded_known():
        return model == m.recorded()
    return requested == m.requested


def _context_length_message(
    e: "llm.ContextLengthError", m: "llm.Model"
) -> str:
    """What to tell the operator when m's server 400s a call as too long.

    The limit in it is the server's own number, read from the body of its
    400 — a measurement, not this run's budget; the budget is what the run
    asked for (m's window). Naming both, and only the setting that fixes
    it, keeps the failure from looking like a model or a network problem."""
    budget = m.ctx_budget
    if e.prompt_tokens is not None and e.prompt_tokens >= e.server_limit:
        return (
            f"\nThe server of {m.label()} rejected a call as too long: the "
            f"prompt alone ({e.prompt_tokens} tokens) already exceeds its "
            f"context window ({e.server_limit} tokens), and this run budgets "
            f"{budget}. No window setting fixes a prompt that no longer fits "
            f"— the DAG has grown past the model's window. Re-run against a "
            f"server launched with a larger --max-model-len, route the role "
            f"to a model with a larger window, or start a fresh DAG for this "
            f"conjecture."
        )
    return (
        f"\nThe server of {m.label()} rejected a call as too long: its "
        f"context window is {e.server_limit} tokens (the server's own "
        f"number, from its 400), but this run budgets {budget}, so every "
        f"call whose prompt plus generation budget outruns the window is "
        f"rejected outright — a rejection the compaction rescue never sees, "
        f"because it only triggers on a truncated reply. To match the "
        f"server's --max-model-len, {m.ctx_budget_fix(e.server_limit)}."
    )


def _fail_context_length(ctx, e: "llm.ContextLengthError",
                         stop: Any = None) -> None:
    """Run-level handling of a server context 400: one loud message per
    run, then the stop the caller names (parallel mode stops the shared
    state; serial mode exits from main instead). The checkpoint is
    untouched — it holds the last safe boundary, which is exactly where a
    re-run with a corrected --ctx-budget should resume from. Logged
    unconditionally: a fatal the operator must see is not verbose noise.
    The message names the model whose server refused (e.model, set by
    _chat), once per model."""
    m = e.model
    said = f"context-length:{m.name}" if m is not None else "context-length"
    if said not in ctx.run.said:
        ctx.run.said.add(said)
        if m is not None:
            ctx.ui.log(_context_length_message(e, m))
        else:
            ctx.ui.log(f"\nThe server rejected a call as too long: {e}")
        ctx.ui.log(
            "The checkpoint is safe; re-running resumes from the last "
            "safe boundary.",
        )
    if stop is not None:
        stop()


def _headroom(m: "llm.Model", messages: List[Dict[str, Any]],
              tools: Optional[List[Dict[str, Any]]] = None) -> int:
    """Tokens m's context budget has left for generation after the prompt
    (messages, and the tools field when one is sent).

    ctx_budget caps prompt + generation together, so this is the most a
    response can ever get. The budget is set so that prompt + generation
    lands on COMPACT_THRESHOLD * ctx_budget: the model may think and write
    until the budget is COMPACT_THRESHOLD full, and only then is it cut off
    for compaction. The prompt is counted by the server where it can
    count (_prompt_tokens), so the sum is exact at any prompt length.
    """
    return _room(_prompt_tokens(m, messages, tools), m.ctx_budget)


def _prompt_tokens(m: "llm.Model", messages: List[Dict[str, Any]],
                   tools: Optional[List[Dict[str, Any]]] = None) -> int:
    """The size in tokens of the prompt messages (and tools, the
    OpenAI-style field) make on m: the server's own count, chat template
    included, where it offers one (Backend.count_tokens); else estimated
    from characters (_estimate_tokens). An estimate's error grows with the
    prompt — LaTeX runs nearer 2.5 characters a token than 4 — so on a
    server that counts, nothing that sizes a call is estimated."""
    counted = m.backend.count_tokens(messages, tools)
    if counted is not None:
        return counted
    estimate = _estimate_tokens(m, messages)
    if tools:
        estimate += int(len(json.dumps(tools))
                        / max(m.backend.chars_per_token(), 1.0))
    return estimate


def _estimate_tokens(m: "llm.Model", messages: List[Dict[str, Any]]) -> int:
    """The size in tokens of a list of messages, estimated from their
    characters and m's running chars/token ratio — the fallback where the
    server counts nothing (_prompt_tokens), and the quick measure of a
    prompt's parts. The ratio starts from the model's calibrated seed and
    is revised from every real count and usage the server reports
    (Backend.note_usage)."""
    # A tool-call turn carries its calls beside an often-empty content, and
    # sometimes its reasoning trace (reason()'s Model.echo_reasoning).
    chars = sum(
        len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or ""))
        + llm.reasoning_chars(m)
        for m in messages
    )
    ratio = m.backend.chars_per_token()
    return int(chars / max(ratio, 1.0)) + 1


def _room(used: int, ctx_budget: int) -> int:
    """The generation budget a prompt of `used` tokens leaves: up to
    COMPACT_THRESHOLD of the window, floored at 0 (see _headroom)."""
    return max(int(ctx_budget * COMPACT_THRESHOLD) - used, 0)


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
    instead of separating it: in <think>...</think>, or — when the chat
    template opened <think> in the prompt and the server ran without a
    reasoning parser — as everything before a closing </think> that has
    no opening tag."""
    if thinking or ("\u003cthink\u003e" not in content
                    and "\u003c/think\u003e" not in content):
        return content, thinking
    m = re.search(r"\u003cthink\u003e(.*?)\u003c/think\u003e", content, re.DOTALL)
    if m:
        thinking = m.group(1).strip()
    elif "\u003c/think\u003e" in content and "\u003cthink\u003e" not in content:
        thinking = content.rsplit("\u003c/think\u003e", 1)[0].strip()
    return strip_thinking(content).strip(), thinking


def _summary_room(ctx_budget: int) -> int:
    """The room a compaction call keeps for its summary, clamped like
    _reserve so a small window still leaves most of itself for the trace."""
    return min(COMPACT_SUMMARY_TOKENS, ctx_budget // 4)


def _compact_messages(part: str, prev: str, tail: str, part_no: int,
                      last: bool) -> List[Dict[str, str]]:
    """The messages of one compaction call over part of the trace."""
    which = "the last part" if last else "more parts follow"
    user = (
        f"Part {part_no} of the cut-off reasoning trace ({which}):\n"
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
    return [
        {"role": "system", "content": COMPACT_SYSTEM},
        {"role": "user", "content": user},
    ]


def _compact_budget(m: "llm.Model", prev: str, tail: str) -> int:
    """Tokens of trace one compaction call on m can take: the window left
    after the call's fixed text (system prompt, framing, the summary so
    far, the answer's tail) and the room kept for the summary it writes."""
    fixed = _estimate_tokens(m, _compact_messages("", prev, tail, 1, False))
    return _room(fixed, m.ctx_budget) - _summary_room(m.ctx_budget)


def _take_lines(lines: List[str], start: int, max_chars: int) -> int:
    """The end of the longest run of lines from `start` that fits max_chars.
    Breaking on newlines keeps no LaTeX display cut mid-line; the run is at
    least one line, so one very long line still moves the split on, and the
    summary's room absorbs the overshoot."""
    end, size = start, 0
    while end < len(lines):
        add = len(lines[end]) + (end > start)
        if end > start and size + add > max_chars:
            break
        size += add
        end += 1
    return end


def _compact_pass(ctx, m: "llm.Model", part: str, prev: str, tail: str,
                  part_no: int, last: bool, verbose: bool) -> str:
    """One summarisation pass over part of the cut-off trace, on m.

    Returns the folded summary, or `prev` unchanged on any failure: the
    compaction must never lose what it already had, and it is a rescue, not a
    new stage that can fail the call — a pass the server refuses as too long
    for its window keeps `prev` too. A spent quota is raised, for
    _compact_trace to take the passes to the fallback.
    """
    messages = _compact_messages(part, prev, tail, part_no, last)
    room = _headroom(m, messages)
    # A chunk is sized to leave _summary_room for the summary (a quarter
    # of a small window); far less than that means the estimate was off
    # and the pass would be cut short or refused.
    if room < min(COMPACT_MIN_ROOM, _summary_room(m.ctx_budget) // 2):
        ctx.ui.log(f"  compaction chunk {part_no} leaves only ~{room} tokens of "
                   f"{m.label()}'s window; keeping the previous summary.",
                   verbose)
        return prev
    options = {**_call_options(m), "num_predict": room}
    try:
        reply = _chat(ctx, m, messages, think=None, schema=None,
                      options=options)
    except llm.ContextLengthError as e:
        ctx.ui.log(f"  compaction chunk {part_no} is too long for "
                   f"{m.label()} ({e}); keeping the previous summary.", verbose)
        return prev
    except (requests.RequestException, ValueError, KeyError) as e:
        ctx.ui.log(f"  compaction chunk {part_no} transport error: {e}",
            verbose)
        return prev
    summary = (reply.content or "").strip()
    if reply.truncated or not summary:
        ctx.ui.log(f"  compaction chunk {part_no} was truncated or "
            f"empty; keeping the previous summary.", verbose)
        return prev
    return summary


def _compact_trace(ctx, role: str, thinking: str, tail: str, verbose: bool,
                   prev_summary: str = "") -> str:
    """Summarise a cut-off reasoning trace in pi-style passes: in one call
    when it fits the window, else in chunks, each sized to what the window
    leaves beside the summary so far and folding that summary in. prev_summary carries an earlier
    compaction round's result across, so a trace compacted in several rounds
    (a continuation that itself hit the wall) never loses what the first
    rounds established. Returns the summary, or prev_summary unchanged when
    the trace was empty or nothing survived the passes. The passes go to
    the model the compact stage is routed to, role's by default."""
    if not thinking.strip():
        return prev_summary
    m = _stage_model(ctx, "compact", role)
    ratio = m.backend.chars_per_token()
    ctx.ui.log(f"  {role} trace (~{int(len(thinking) / ratio)} tokens) too "
        f"big to re-send; summarising it.", verbose)
    lines = thinking.split("\n")
    summary = prev_summary
    start, part_no = 0, 0
    while start < len(lines):
        # Sized per chunk: the summary folded into the next call grows.
        budget = _compact_budget(m, summary, tail)
        end = _take_lines(lines, start, max(int(budget * ratio), 1000))
        part = "\n".join(lines[start:end])
        part_no += 1
        last = end == len(lines)
        ctx.ui.log(f"  compaction chunk {part_no}: ~{int(len(part) / ratio)} "
            f"tokens of trace{' (last)' if last else ''}.", verbose)
        try:
            summary = _compact_pass(ctx, m, part, summary, tail, part_no,
                                    last, verbose)
        except llm.QuotaExhausted:
            # The compaction model's quota used up: the rest of the trace
            # goes to the fallback (--with-fallback), chunked again from
            # here for its window, or the refusal goes up to the command.
            fallback = _fallback_for(ctx, m)
            if fallback is None:
                raise
            ctx.ui.log(f"  compaction: {m.label()}'s quota is used up; "
                       f"summarising the rest on the fallback, "
                       f"{fallback.label()}.", verbose)
            m, ratio = fallback, fallback.backend.chars_per_token()
            part_no -= 1
            continue
        start = end
    return summary


def compaction_passes(ctx, noun: str = "compaction passes") -> str:
    """The passes an overflowed call was given, as a message names them:
    "2 compaction passes", or "its compaction passes" with no limit."""
    n = ctx.settings.max_compaction_passes
    return f"its {noun}" if n < 0 else f"{n} {noun}"


def _resume_compacted(
    ctx,
    role: str,
    system_prompt: str,
    user_prompt: str,
    thinking: str,
    content: str,
    verbose: bool,
    max_passes: Optional[int] = None,
) -> Tuple[str, Optional[Dict[str, str]]]:
    """Pi-style overflow recovery after a truncated call.

    One pass: compact the reasoning trace (the scratch, summarised), keep
    the task prompt and the answer-so-far verbatim (the recent work), and
    ask the model to resume from the cut with the whole headroom of the
    now-smaller prompt: P + summary + C is strictly less than P + trace + C
    was.

    A continuation that hits the wall is not a failure but the next pass: its
    trace joins the summary, its content extends the verbatim answer, and the
    model resumes from the new cut. The summary carries across passes, so
    nothing established earlier is lost.

    Returns (answer, None) once a pass completes the answer, or ("", state)
    when the work is still unfinished after `max_passes` passes (or a pass can
    no longer go on). `state` is {"summary", "thinking", "answer"}, the
    partial work for the caller to hand on — the proof loop compacts it a
    final time and sends it to the reviser to pick a smaller lemma or
    revise the statement.
    """
    if max_passes is None:
        max_passes = ctx.settings.max_compaction_passes
    m = _model_for(ctx, role)
    answer = content
    summary = ""

    def _partial() -> Dict[str, str]:
        # The work-so-far for a caller to hand on: the running summary, the
        # latest (not-yet-compacted) trace, and the verbatim answer.
        return {"summary": summary, "thinking": thinking, "answer": answer}

    for _pass in settings_mod.counts(max_passes, start=1):
        ctx.ui.log(f"  {role} compaction pass "
            f"{settings_mod.of_limit(_pass, max_passes)}.", verbose)
        tail = answer[-COMPACT_TAIL_CHARS:]
        summary = _compact_trace(ctx, role, thinking, tail, verbose, summary)

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
        # Show where the continuation prompt's tokens go: the original task is
        # re-sent in full, then the compaction (trace summary + verbatim
        # answer-so-far) rides on top. The two are different fixes, so keep
        # them separate in the log.
        _ratio = m.backend.chars_per_token()
        ctx.ui.log(
            f"  {role} continuation prompt: task~{int(len(user_prompt)/_ratio)}"
            f" + summary~{int(len(summary)/_ratio)}"
            f" + answer~{int(len(answer)/_ratio)} tokens",
            verbose,
        )
        room = _headroom(m, messages)
        if room < COMPACT_MIN_ROOM:
            ctx.ui.log(f"  {role} compaction left only ~{room} tokens of room; "
                f"not enough to continue.", verbose)
            return "", _partial()
        options = {
            **_call_options(m),
            "num_predict": room,
        }
        try:
            reply = _chat(ctx, m, messages, think=True, schema=None,
                          options=options)
        except (requests.RequestException, ValueError, KeyError) as e:
            ctx.ui.log(f"  {role} continuation transport error: {e}", verbose)
            return "", _partial()
        cont_content, cont_thinking = _split_inline_thinking(
            reply.content or "", reply.thinking or "")
        if reply.truncated:
            if not (cont_content.strip() or cont_thinking.strip()):
                ctx.ui.log(f"  {role} continuation was truncated with nothing "
                    f"to build on; the answer does not fit the window.",
                    verbose)
                return "", _partial()
            # The continuation itself hit the wall: not a failure, the next
            # pass. Its trace joins the summary, its content extends the
            # verbatim answer, and the model resumes from the new cut.
            answer += cont_content
            thinking = cont_thinking
            ctx.ui.log(f"  {role} continuation hit the wall; the next pass "
                f"compacts its trace and resumes from the new cut.",
                verbose)
            continue
        if not cont_content.strip():
            ctx.ui.log(f"  {role} continuation returned empty content.", verbose)
            return "", _partial()
        return (answer + cont_content).strip(), None
    # All `max_passes` passes spent and the answer is still unfinished.
    ctx.ui.log(
        f"  {role} still unfinished after {max_passes} compaction passes; "
        f"handing the partial work on.",
        verbose,
    )
    return "", _partial()


def _incomplete_report(
    ctx,
    role: str, partial: Dict[str, str], verbose: bool,
) -> str:
    """The third compaction, for a prover that overflowed the window.

    The two bounded compaction passes left the last (cut-off) trace not yet
    summarised; fold it into the running summary the way a continuation pass
    would have, and pair the summary with the verbatim partial answer. The
    result is what the reviser reads to pick a smaller lemma or revise the
    statement. Returns the report text, or "" when there is nothing to
    report.
    """
    thinking = str(partial.get("thinking") or "")
    answer = str(partial.get("answer") or "")
    if not (thinking.strip() or answer.strip()):
        return ""
    summary = str(partial.get("summary") or "")
    summary = _compact_trace(
        ctx,
        role, thinking, answer[-COMPACT_TAIL_CHARS:], verbose, summary,
    )
    parts = [
        "The prover ran out of its context window and did not finish: after "
        f"{compaction_passes(ctx, 'compaction-and-resume passes')} the proof was "
        "still incomplete. Read its work below and choose a smaller lemma it "
        "should try instead — or a corrected statement, if its work shows "
        "the statement is false or badly posed.",
    ]
    if summary.strip():
        parts += [
            "Summary of its reasoning trace (what it established, where it "
            "stopped, what it cited):",
            summary,
        ]
    if answer.strip():
        parts += [
            "The partial answer it wrote, verbatim, ending exactly where it "
            "was cut off:",
            answer,
        ]
    return "\n\n".join(parts)


# How the status board names a role (Context.status): the agent's prompt
# name, without the parallel-mode prefixes.
_ROLE_LABELS = {
    "parallel_planner": "planner",
    "parallel_lemma_generator": "generator",
}


def role_label(role: str) -> str:
    return _ROLE_LABELS.get(role, role.replace("_", " "))


def reason(
    ctx,
    system_prompt: str,
    user_prompt: str,
    role: str,
    verbose: bool = True,
    tools: Optional["tools_mod.ToolSet"] = None,
) -> Tuple[str, str, Optional[Dict[str, str]]]:
    """_reason, with the tools it was offered closed when the call ends
    (ToolSet.close: a reviser's or a verifier's scripts go with it), and
    the fallback (--with-fallback) taken when the role's model cannot
    serve the call: its quota is used up (llm.QuotaExhausted, at once), or
    it never answered through the retries ("unavailable"). Per call: the
    next call goes to the role's own model first.

    The call is sent again from the start, pinned to the fallback. Its
    prompt is fitted again for the fallback's window when it is the one
    _fit_prompt last fitted for the role on this thread; otherwise it is
    sent as it is, and "too_large" if it does not fit. A verifier's call
    is not sent on: VerifierFailover restarts its whole check on the
    fallback, since a certificate names one model. With no fallback — or
    the role already on it — a spent quota is raised to the command and a
    silent server is "unavailable", as without the flag."""
    toolsets = [tools]
    try:
        m = _model_for(ctx, role)
        try:
            result = _reason(ctx, system_prompt, user_prompt, role, verbose, tools)
        except llm.QuotaExhausted as e:
            # Only a refusal by the role's own model fails the call over:
            # a stage's model (compaction) fails over by itself, and one
            # that reaches here had nothing to fall back to.
            fallback = _fallback_for(ctx, m) if e.model in (None, m) else None
            if fallback is None:
                raise
            why = f"its quota is used up ({e})"
        else:
            fallback = (_fallback_for(ctx, m) if result[1] == "unavailable"
                        else None)
            if fallback is None:
                ctx.loop.served[role] = m
                return result
            why = "it did not answer through its retries"
        if role in VERIFIER_ROLES:
            raise VerifierFailover(fallback, f"{m.label()}: {why}")
        ctx.ui.log(f"  {role}: {m.label()} cannot serve the call ({why}); "
                   f"sending it to the fallback, {fallback.label()}.", verbose)
        with pinned(ctx, {role: fallback}):
            fitted = ctx.loop.fitted.get(role)
            if fitted is not None and fitted[0] == user_prompt:
                user_prompt, tools = fitted[1]()
                toolsets.append(tools)
            result = _reason(ctx, system_prompt, user_prompt, role, verbose, tools)
        ctx.loop.served[role] = fallback
        return result
    finally:
        for toolset in toolsets:
            if toolset:
                toolset.close()


def _reason(
    ctx,
    system_prompt: str,
    user_prompt: str,
    role: str,
    verbose: bool = True,
    tools: Optional["tools_mod.ToolSet"] = None,
) -> Tuple[str, str, Optional[Dict[str, str]]]:
    """Stage 1: free-form reasoning. No `format`, so thinking is preserved.

    Returns (content, status, partial). status is "" on success, or "ceiling"
    when the role exhausted the context window without finishing — a signal
    that the task is too large, not that the call failed — or "unavailable"
    when every attempt failed to reach the server (a transport error, never
    a reply): an outage, not the model's failure, which the proof loop
    waits out instead of counting against the lemma, or "too_large" when
    the prompt itself leaves less than COMPACT_MIN_ROOM of the window for
    the reply, so it was never sent (a request for max_tokens 0 is one a
    server refuses): the same prompt can never succeed, and the caller must
    not send it again unchanged. A model that replied
    with nothing, every attempt, is "" with empty content. `partial` is None on
    success; on "ceiling" after a bounded compaction rescue it carries the
    partial work ({"summary", "thinking", "answer"}) for the caller to hand
    on — the proof loop compacts it a final time and sends it to the reviser.

    Truncated output is never returned as if it were complete: `done_reason ==
    "length"` means the model was cut off mid-sentence, and the budget already
    covers the whole window. Before a truncation is reported as "ceiling" the
    exchange gets a pi-style compaction rescue — the thinking trace
    summarised, the answer written so far kept verbatim, the model asked to
    resume from the cut (see _resume_compacted). The rescue is bounded by
    llm.max_compaction_passes passes: a continuation that is itself truncated
    becomes the next pass's input, and once the passes are spent (or a pass
    leaves nothing to build on, or no headroom is left) the partial work is
    returned for the caller to decide what to do — for the prover, handing it
    to the reviser to pick a smaller lemma or revise the statement.

    tools, when given, is what the agent may call before it answers
    (tools/): natively, through the request's `tools` field, or — on a
    server that refuses that, or under --tool-mode json — through the JSON
    protocol the system prompt then describes. Each round's results stay in
    the conversation, up to the ToolSet's max_rounds() rounds; then a native
    conversation is folded (below) into one plain message with the results
    written in, and a JSON-protocol one is told to answer. A native request
    never declares tools it does not offer: with tool_choice "none" a model
    that still wants a lookup writes the call anyway, a vLLM server drops
    it, and the reply comes back empty. A compaction rescue re-sends the task with the results
    folded into it as text, so what the agent looked up survives the cut.

    Lookups never use the reply's reserve (_reserve: RESERVE_TOKENS, or a
    quarter of a small window). The prompt's size is the server's own count
    of the last prompt sent (usage.prompt_tokens) plus an estimate of what
    was appended since, or the character estimate alone before any count
    came back. Once the prompt reaches the reserve the lookups close: a
    native call is folded (below) into one plain message that declares no
    tools, and under the JSON protocol the last result says it was the
    last; a prompt that starts there is offered no tools at all. A result that would cut into the reserve is
    trimmed to the room left (tools.ToolSet.run), and every result ends
    with a line telling the model how much room remains.

    The tool conversation is the one part of a call whose shape servers and
    models disagree on, so when it fails after a tool round (an empty
    reply, the server refusing it) it is rescued in two steps, each tried
    once. First it is sent again with the reasoning traces switched: added
    to each tool-call turn, for a thinking model that needs to see them
    again (remembered for the run once that works, Model.echo_reasoning), or
    dropped, if they were being sent. Then it is folded into one plain
    request: the task with the tool results written into it and no tools
    offered, the form every chat server and model takes. A reply that keeps
    calling closed tools is folded at once (a safety net: native lookups
    are folded as they close). A call whose tool conversation
    works is sent exactly as it would be without the rescue.
    """
    ctx.status(role=role_label(role))
    # The model routed to this role (llm/routing.py): every request of the
    # call, and the budgets they are sized by, are its.
    m = _model_for(ctx, role)
    base_system = system_prompt
    # The tools as offered, for the fold notes of a task re-sent without
    # them (tools itself is dropped when they close).
    offered_tools = tools
    ctx_budget = m.ctx_budget
    reserve = _reserve(ctx_budget)
    native = bool(tools) and _native_tools(ctx, m)
    if tools and not native:
        system_prompt = base_system + "\n\n" + tools.json_protocol()
    # user_prompt is the task without tools: the tools' hints ride on it
    # only while the tools are offered, and every request that offers none
    # (closed from the start, a fold, a compaction rescue) is built from
    # user_prompt alone, so it never tells the model to call a tool.
    hints = tools.prompt_hints() if tools else ""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": user_prompt + ("\n\n" + hints if hints else "")},
    ]
    # closed: lookups have been closed for lack of room, and said so (once
    # per call).
    closed = False

    def _no_lookup_room(room: int) -> bool:
        """Whether a prompt leaving `room` for the reply has no room for
        another lookup result before the reserve."""
        return room - reserve - _RESULT_FRAME_TOKENS < _MIN_RESULT_TOKENS

    if tools and _no_lookup_room(_headroom(
            m, messages, tools.specs() if native else None)):
        # The task alone already reaches into the reply's reserve: no
        # lookup could fit, so none is offered — not the tools field, not
        # the protocol in the system prompt.
        closed = True
        tools = None
        native = False
        system_prompt = base_system
        messages[:] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        room = _room(_estimate_tokens(m, messages), ctx_budget)
        if room >= COMPACT_MIN_ROOM:   # else the loop's too_large says it
            ctx.ui.log(f"  {role}: closing lookups, ~{room:,} tokens left for the "
                f"answer.", verbose)
    # offered: the model was offered tools in this call, so a request that
    # takes them away says so. max_rounds: the rounds it may spend.
    offered = bool(tools)
    max_rounds = tools.max_rounds() if tools else ctx.settings.max_tool_rounds
    options: Dict[str, Any] = _call_options(m)
    # The tool calls made and their results, as text: what a compaction
    # rescue folds into the task it re-sends.
    tool_notes: List[str] = []
    rounds = 0
    # The reasoning trace of each native tool-call turn, by the turn's index
    # in messages: sent with it only while echo is on.
    traces: Dict[int, Dict[str, Any]] = {}
    echo = m.echo_reasoning
    # switched: the traces have been switched on or off this call, the
    # rescue's first step; probing: switched on, and not yet shown to help.
    # folded: the tool conversation has been folded, its second.
    switched = probing = folded = False
    # The server's count of the last prompt it was sent, and how many
    # messages that prompt held: the next prompt in this call is that count
    # plus an estimate of only what was appended since (_used). None before
    # any count came back, and again whenever the conversation is reshaped
    # rather than appended to (a fold, the traces switched, the JSON
    # protocol taking over), since the count no longer describes a prefix.
    measured: Optional[Tuple[int, int]] = None

    def _used(outgoing: List[Dict[str, Any]]) -> int:
        if measured is None:
            return _estimate_tokens(m, outgoing)
        tokens, sent = measured
        return tokens + (_estimate_tokens(m, outgoing[sent:])
                         if len(outgoing) > sent else 0)

    def _spare() -> int:
        """Tokens the prompt as it now stands has left before the reserve."""
        return _room(_used(_outgoing() if native else messages),
                     ctx_budget) - reserve

    def _result_budget() -> int:
        """The characters the next tool result may take: the room left
        before the reserve, less the result's framing."""
        return int((_spare() - _RESULT_FRAME_TOKENS)
                   * m.backend.chars_per_token())

    def _room_line() -> str:
        """The line every tool result ends with: the room the prompt has
        left before lookups close, once this result is in it."""
        left = _spare() - _RESULT_FRAME_TOKENS
        if left < _MIN_RESULT_TOKENS:
            return ("[No room remains for more lookups: give your answer "
                    "now, with what you have.]")
        return (f"[About {left // 100 * 100:,} tokens of room remain before "
                f"lookups close.]")

    def _outgoing() -> List[Dict[str, Any]]:
        if not (echo and traces):
            return messages
        return [{**m, **traces[i]} if i in traces else m
                for i, m in enumerate(messages)]

    def _task_without_tools(then: str) -> str:
        """The task as a request that offers no tools re-sends it: without
        the tools' hints, with the results of the calls made so far and
        what they led to (ToolSet.fold_notes: the agent's files), and —
        when tools were offered in this call — a line saying they are
        closed, ending with `then`."""
        task = user_prompt
        if tool_notes:
            task += ("\n\n---\nResults of the tool calls you made:\n\n"
                     + "\n\n".join(tool_notes))
            notes = offered_tools.fold_notes() if offered_tools else ""
            if notes:
                task += "\n\n---\n" + notes
        if offered:
            task += f"\n\n---\nThe tools are closed now: {then}"
        return task

    def _fold(why: str) -> None:
        nonlocal native, folded, probing, measured
        task = _task_without_tools(
            "give your answer, in the format asked for.")
        messages[:] = [
            {"role": "system", "content": base_system},
            {"role": "user", "content": task},
        ]
        traces.clear()
        native = False
        folded = True
        probing = False
        measured = None
        ctx.ui.log(f"  {role} {why}; asking in one plain message with its "
            f"{len(tool_notes)} tool result(s) written in.", verbose)

    def _rescue(why: str) -> bool:
        """Take the rescue's next step for a failed tool conversation.
        False when there is none left, or no tool round to blame."""
        nonlocal echo, switched, probing, measured
        if folded or not tool_notes:
            return False
        if traces and not switched:
            switched = True
            echo = probing = not echo
            measured = None
            ctx.ui.log(f"  {role} {why}; sending its tool conversation again "
                f"{'with' if echo else 'without'} its reasoning traces.",
                verbose)
            return True
        _fold(why)
        return True

    def _record(name: str, arguments: Any, result: str) -> None:
        shown = arguments if isinstance(arguments, str) else json.dumps(arguments)
        ctx.ui.log(f"  {role} called {name}({shown[:200]})", verbose)
        tool_notes.append(f"{name}({shown}) returned:\n{result}")

    attempt = 0
    # Whether the server ever answered this call: if not, the empty result
    # is an outage ("unavailable"), not the model's.
    answered = False
    while settings_mod.below(attempt, ctx.settings.llm_max_retries):
        offer = (bool(tools) and not folded
                 and settings_mod.below(rounds, max_rounds))
        outgoing = _outgoing() if native else messages
        if native and not offer:
            # The last round of lookups is spent. Never a request that
            # declares tools with tool_choice "none": a model that still
            # wants a lookup writes the call anyway, a server (vLLM) drops
            # it, and the reply comes back empty. One plain message with
            # the results written in declares no tools at all.
            _fold(f"has used its {max_rounds} rounds of "
                  f"lookups")
            continue
        # As much room as the budget has left, recomputed for this prompt:
        # the prompt differs every iteration and every role — and grows
        # with each tool round — and there is no point asking for more than
        # the budget holds or less than it does. The server counts the
        # prompt exactly, tools field included, where it can; that count
        # is then what _used builds on until the conversation is reshaped.
        counted = m.backend.count_tokens(
            outgoing, tools.specs() if native else None)
        if counted is not None:
            measured = (counted, len(outgoing))
        room = _room(_used(outgoing), ctx_budget)
        if native and _no_lookup_room(room):
            # The prompt has reached the reply's reserve: lookups close the
            # same way, by folding, and the model answers with what it
            # has looked up.
            _fold(f"has reached the reply's reserve (~{room:,} tokens left)")
            continue
        if room < COMPACT_MIN_ROOM:
            if tool_notes and not folded:
                # The tool conversation's own weight (its turns, any traces
                # sent back) is what is left to drop.
                _fold(f"left only ~{room:,} tokens for its answer")
                continue
            # Never a request for (next to) nothing: a server refuses
            # max_tokens 0, and a thinking model cannot answer in less.
            ctx.ui.log(f"  {role}: the prompt leaves only ~{room:,} tokens of the "
                f"{ctx_budget:,}-token window for the reply; not sending it.",
                verbose)
            return "", "too_large", None
        if offer and _no_lookup_room(room):
            # The JSON protocol at the reply's reserve: no call is served
            # from here, and the last result already said so (its "last
            # tool call" line). There is no tool_choice to get wrong here.
            offer = False
            if not closed:
                closed = True
                ctx.ui.log(f"  {role}: closing lookups, ~{room:,} tokens left for "
                    f"the answer.", verbose)
        options["num_predict"] = room
        try:
            if native:
                # Always offering: a native call whose lookups are closed
                # has been folded above, and sends no tools.
                reply = _chat(
                    ctx, m,
                    outgoing, think=True, schema=None, options=options,
                    tools=tools.specs(),
                )
            else:
                reply = _chat(ctx, m, messages, think=True, schema=None,
                              options=options)
        except llm.ToolsUnsupported as e:
            # The server takes no native tools: the JSON protocol from here
            # on, for this call and every later one. Not an attempt spent.
            _disable_native_tools(ctx, m, str(e), verbose)
            native = False
            measured = None
            system_prompt = base_system + "\n\n" + tools.json_protocol()
            messages[0] = {"role": "system", "content": system_prompt}
            continue
        except requests.HTTPError as e:
            # The server is up and refused this conversation: change the
            # conversation rather than wait for the server.
            status = e.response.status_code if e.response is not None else None
            if (status in _REJECTED_STATUSES
                    and _rescue(f"conversation was refused (HTTP {status})")):
                answered = True
                continue
            attempt += 1
            ctx.ui.log(f"  {role} transport error (attempt {attempt}): {e}", verbose)
            if settings_mod.below(attempt, ctx.settings.llm_max_retries):
                _retry_pause(ctx, attempt, e, role, verbose)
            continue
        except (requests.RequestException, ValueError, KeyError) as e:
            attempt += 1
            ctx.ui.log(f"  {role} transport error (attempt {attempt}): {e}", verbose)
            if settings_mod.below(attempt, ctx.settings.llm_max_retries):
                _retry_pause(ctx, attempt, e, role, verbose)
            continue

        answered = True
        prompt_tokens = reply.usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            measured = (prompt_tokens, len(outgoing))
        content = reply.content or ""
        thinking = reply.thinking or ""
        content, thinking = _split_inline_thinking(content, thinking)
        hit_ceiling = reply.truncated
        if probing and (reply.tool_calls or content.strip()):
            probing = False
            if not m.echo_reasoning:
                m.echo_reasoning = True
                ctx.ui.log(f"{m.label()} needs its reasoning sent back with "
                    f"its tool calls; sending it for the rest of the run.",
                    verbose)

        if native and reply.tool_calls and not hit_ceiling:
            if offer:
                rounds += 1
                if reply.reasoning_field:
                    traces[len(messages)] = reply.reasoning_field
                messages.append({
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": c.arguments}}
                        for c in reply.tool_calls
                    ],
                })
                for c in reply.tool_calls:
                    # Each result is held to the room left once the ones
                    # before it in this round are in.
                    result = tools.run(c.name, c.arguments,
                                       budget=_result_budget())
                    _record(c.name, c.arguments, result)
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": result})
                    messages[-1]["content"] = result + "\n\n" + _room_line()
                continue
            if not content.strip():
                # Still calling with the tools closed and nothing written:
                # re-sending the same conversation tends to get the same.
                # A safety net only: a native request is folded once its
                # lookups close, so no request reaches here with tools
                # declared and not offered.
                attempt += 1
                _fold(f"kept calling tools after its last round "
                      f"(attempt {attempt})")
                continue

        if not native and tools and not folded and not hit_ceiling:
            call = tools.parse_json_call(content)
            if call is not None:
                messages.append({"role": "assistant", "content": content})
                if offer:
                    rounds += 1
                    name, arguments = call
                    result = tools.run(name, arguments,
                                       budget=_result_budget())
                    _record(name, arguments, result)
                    messages.append({"role": "user", "content": result})
                    # The room line and the last-call test both read the
                    # prompt with this result in it.
                    line = _room_line()
                    last = (not settings_mod.below(rounds, max_rounds)
                            or _spare() - _RESULT_FRAME_TOKENS
                            < _MIN_RESULT_TOKENS)
                    messages[-1]["content"] = (
                        f"Result of {name}:\n{result}\n\n{line}\n\n"
                        + ("That was your last tool call: give your answer "
                           "now, in the format asked for."
                           if last else
                           "Call another tool, or give your answer in the "
                           "format asked for.")
                    )
                else:
                    attempt += 1
                    messages.append({"role": "user", "content": (
                        "No more tool calls: give your answer now, in the "
                        "format asked for."
                    )})
                continue

        if content.strip() and not hit_ceiling:
            return content.strip(), "", None

        if hit_ceiling:
            spent = (reply.usage.get("completion_tokens")
                     or (len(thinking) + len(content)) // 4)
            ctx.ui.log(
                f"  {role} hit the context wall "
                f"(ctx_budget={options['ctx_budget']}, generated {spent} tokens, "
                f"~{len(thinking) // 4} of them thinking).",
                verbose,
            )
            # A pi-style compaction rescue, when there is something to
            # resume from: the trace gets summarised, the answer-so-far
            # is kept verbatim, and the model is asked to finish from the
            # cut. The rescue is bounded by llm.max_compaction_passes passes;
            # if the work is still unfinished then, its partial state is
            # returned so the proof loop can hand it to the reviser. The
            # continuation offers no tools; what was looked up rides along
            # in the task.
            partial: Optional[Dict[str, str]] = None
            if COMPACT_ENABLED and (thinking.strip() or content.strip()):
                task = _task_without_tools(
                    "make no more tool calls, and finish with what you have.")
                resumed, partial = _resume_compacted(
                    ctx,
                    role, base_system, task, thinking, content, verbose,
                )
                if resumed:
                    ctx.ui.log(f"  {role} completed after compaction + "
                        f"continuation.", verbose)
                    return resumed, "", None
            return "", "ceiling", partial

        attempt += 1
        if not _rescue(f"returned empty content (attempt {attempt})"):
            ctx.ui.log(f"  {role} returned empty content (attempt {attempt}).", verbose)

    # Retries exhausted on empty replies or transport errors; nothing to salvage.
    return "", ("" if answered else "unavailable"), None


def extract(
    ctx,
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

    The call goes to the model the extract stage is routed to, role's by
    default (llm/routing.py). With --with-fallback it is sent once more to
    the fallback when that model's quota is used up or it never answered;
    an extraction records nothing, so it needs no restart of anything.
    """
    m = _stage_model(ctx, "extract", role)
    try:
        result, answered = _extract(ctx, m, instruction, source_text, schema,
                                    role, verbose)
    except llm.QuotaExhausted as e:
        fallback = _fallback_for(ctx, m)
        if fallback is None:
            raise
        why = f"its quota is used up ({e})"
    else:
        fallback = None if answered else _fallback_for(ctx, m)
        if fallback is None:
            return result
        why = "it did not answer through its retries"
    ctx.ui.log(f"  {role} extraction: {m.label()} cannot serve it ({why}); "
               f"sending it to the fallback, {fallback.label()}.", verbose)
    # The source fitted the extraction model's window, not necessarily the
    # fallback's: one that leaves no room for the copy is not sent.
    room = _headroom(fallback, [{"role": "user", "content": source_text}])
    if room < COMPACT_MIN_ROOM:
        ctx.ui.log(f"  {role} extraction: the text leaves only ~{room} tokens "
                   f"of {fallback.label()}'s window; not sending it.", verbose)
        return {}
    return _extract(ctx, fallback, instruction, source_text, schema, role,
                    verbose)[0]


def _extract(
    ctx, m: "llm.Model", instruction: str, source_text: str,
    schema: Dict[str, Any], role: str, verbose: bool,
) -> Tuple[Dict[str, Any], bool]:
    """extract() on m: (the object, whether the server ever answered)."""
    answered = False
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
    use_schema = not m.schema_mode_broken

    for attempt in settings_mod.counts(ctx.settings.llm_max_retries, start=1):
        options = _call_options(m)
        greedy = m.backend.accepts_sampling and not m.greedy_refused
        if greedy:
            options.update(_GREEDY_OPTIONS)
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
        options["num_predict"] = _headroom(m, call_messages)

        content: Optional[str] = None
        try:
            # think is left as None throughout: whichever server we are on,
            # this stage wants the grammar honoured, and the backend knows
            # what that costs. No new mathematics happens here.
            try:
                reply = _chat(
                    ctx, m,
                    call_messages,
                    think=None,
                    schema=schema if use_schema else None,
                    options=options,
                )
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if not (greedy and status in _REJECTED_STATUSES):
                    raise
                # The server refused the request, perhaps for the greedy
                # fields (a reasoning model that takes no temperature):
                # sent again without them, in this same attempt, and never
                # sent to this model again. Greedy sampling is a
                # nicety; it never costs the extraction.
                m.greedy_refused = True
                ctx.ui.log(f"  {role} extraction: {m.label()} refused the "
                           f"request (HTTP {status}); sending it without the "
                           f"greedy sampling fields, for the rest of the run.",
                           verbose)
                options = {k: v for k, v in options.items()
                           if k not in _GREEDY_OPTIONS}
                reply = _chat(
                    ctx, m,
                    call_messages,
                    think=None,
                    schema=schema if use_schema else None,
                    options=options,
                )
            answered = True
            content = reply.content or ""
            if reply.truncated:
                ctx.ui.log(f"  {role} extraction hit the token ceiling.", verbose)

            if use_schema and not content.strip():
                # Grammar/thinking conflict: qwen3.x thinks by default, the
                # format grammar suppresses the answer, and the output lands
                # in the thinking channel. Deterministic, so don't retry
                # same-mode.
                stranded = len(reply.thinking or "")
                ctx.ui.log(
                    f"  {role} extraction returned empty content with format "
                    f"set ({stranded} chars stranded in thinking). Falling back "
                    f"to prompt-only JSON for the rest of the run.",
                    verbose,
                )
                use_schema = False
                m.schema_mode_broken = True
                continue

            parsed = _loads(clean_json_text(
                content, expect=schema.get("required")))
            if not isinstance(parsed, dict):
                # A bare value (a string, a list of strings) is no answer
                # to an object schema; every caller reads the result with
                # .get. Same as a failed extraction: nothing usable.
                ctx.ui.log(
                    f"  {role} extraction returned a JSON "
                    f"{type(parsed).__name__}, not an object; treating it "
                    f"as no answer.",
                    verbose,
                )
                return {}, True
            required = [k for k in schema.get("required") or ()]
            if required and not any(k in parsed for k in required):
                # An object, but not the answer: none of the fields asked
                # for. Asked again, like an unreadable reply.
                ctx.ui.log(f"  {role} extraction returned an object without "
                           f"the fields asked for (attempt {attempt}).", verbose)
                messages = messages[:2] + [
                    {"role": "assistant", "content": content[:1500]},
                    {"role": "user", "content": (
                        f"That object has none of the fields asked for "
                        f"({', '.join(required)}). Reply with ONLY the JSON "
                        f"object, with those fields, no fences and no "
                        f"commentary.")},
                ]
                continue
            return parsed, True
        except (requests.RequestException, KeyError, TypeError) as e:
            ctx.ui.log(f"  {role} extraction transport error (attempt {attempt}): {e}", verbose)
            if settings_mod.below(attempt, ctx.settings.llm_max_retries):
                _retry_pause(ctx, attempt, e, f"{role} extraction", verbose)
        except json.JSONDecodeError as e:
            ctx.ui.log(f"  {role} extraction parse error (attempt {attempt}): {e}", verbose)
            if content is not None:
                ctx.ui.log(f"     raw ({len(content)} chars) head: {content[:200]!r}", verbose)
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

    ctx.ui.log(f"  {role} extraction failed after all retries.", verbose)
    return {}, answered


def _wait_for_server(ctx, role: str, should_stop, verbose: bool) -> bool:
    """Wait SERVER_WAIT_SECONDS after a call the server never answered.
    True when the caller should retry, False when should_stop fired
    during the wait (the engine then returns its published round as
    stopped). Logged unconditionally: a stalled run must say why."""
    ctx.ui.log(
        f"The server did not answer the {role}; this round is not "
        f"counted. Retrying in {SERVER_WAIT_SECONDS}s (Ctrl-C stops the "
        f"run, and a re-run resumes this round)."
    )
    ctx.status(event=f"server unreachable; retrying the {role_label(role)}")
    deadline = time.monotonic() + SERVER_WAIT_SECONDS
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            return False
        time.sleep(1)
    return True


def _reason_until_answered(ctx, should_stop, verbose: bool, *args, **kwargs):
    """reason(), retried while the server is unreachable: its result once
    the server answers, or None when should_stop fired while waiting. A
    "too_large" result is returned at once: it was never sent, so there is
    no outage to wait out."""
    role = args[2]
    while True:
        result = reason(ctx, *args, verbose=verbose, **kwargs)
        if result[1] != "unavailable":
            return result
        if not _wait_for_server(ctx, role, should_stop, verbose):
            return None
