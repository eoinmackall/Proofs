"""The LLM backend: the one place in this project that knows how to talk to a
model over HTTP, and how much headroom to leave it.

The design goal is to make "a model" swappable without touching any of the
proof machinery. main.py's reason() and extract() never import requests, never
build a payload, and never know which model is loaded. They call a small
surface here:

    BACKEND.chat(messages, think=..., schema=..., options={...}) -> Reply

Everything model-specific has to live in exactly one of two places, in order
of preference:

  * A probe  — read off the server at startup, so it is correct for whatever
    is actually loaded, no matter what the alias is: the context ceiling, the
    thinking support, and — on a llama-server — the file name of the model
    it has loaded, from /props. See probe(). Where there is no /props, the
    model that answered is read off each reply instead (recorded_model()).
  * Transport  — HOW to talk to the server. Real code. There is exactly one:
    the OpenAI-compatible /v1/chat/completions endpoint. llama.cpp's
    llama-server serves it locally; a hosted API (OpenAI, OpenRouter,
    DeepSeek, ...) serves it at its base URL. The same transport covers both.
    The only difference is that a hosted API wants an
    `Authorization: Bearer <key>` header, which this module sends whenever an
    API key is supplied (--api-key or $LLM_API_KEY).

The dialect of that endpoint — which sampling fields it accepts, and whether
schema calls must have thinking suppressed — is a probe decision by default
(/props answers => llama.cpp), overridable with --backend
{auto,llamacpp,openai} for the cases the probe can't see.

The thing to avoid is a third axis of `if model.startswith("qwen3.8")`
branches scattered through reason()/extract(). Everything genuinely
model-specific in here is probed; there is no hand-maintained model table, so
a model you have not seen before runs on its own card and its defaults.

Usage from main.py:

    BACKEND = llm_backend.make_backend(args.model, args.host,
                                       api_key=args.api_key,
                                       backend=args.backend)
    PROFILE = BACKEND.probe()
    if PROFILE.auth_error:            # a supplied key the API rejected
        raise SystemExit(PROFILE.auth_error)
    reply = BACKEND.chat(messages, think=..., schema=None, options={...})
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# What we know about a model
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    """What we know about the loaded model. Probed where possible."""
    name: str
    # The file name of the model a llama-server has loaded, read off /props
    # by probe(). None on an endpoint with no /props. Kept apart from name,
    # which is what --model asked for: a llama-server ignores the request's
    # model field and answers with whatever it loaded.
    model_name: Optional[str] = None
    context_limit: int = 65536     # ceiling of the server's KV cache (tokens).
                                   # --num-ctx must fit inside this.
    supports_thinking: bool = False
    probe_ok: bool = False         # False => everything below is a guess
    # Transport facts, set by probe(). is_llama_cpp decides the option set;
    # key_verified / auth_error report how a supplied API key was received.
    # backend is what --backend asked for, probed_llama_cpp what the /props
    # probe found, is_llama_cpp the two reconciled — chat() reads only that.
    backend: str = "auto"
    probed_llama_cpp: bool = False
    is_llama_cpp: bool = False
    key_verified: bool = False
    auth_error: Optional[str] = None
    # Per-model sampler values, empty by default: no model table ships with
    # this code, so this stays {} unless something fills it. REASONING_OPTIONS
    # in main.py (num_predict etc.) are layered on top of these by the backend.
    sampling: Dict[str, Any] = field(default_factory=dict)
    # Calibration seed, not capability: how many characters of prompt text
    # one token of context roughly buys. probe() uses it to seed the running
    # estimate (_RUNNING_CHARS_PER_TOKEN), which note_usage() revises from
    # each call's real usage; estimate_tokens() and main.py's char-based
    # budgets read that running value, not this seed.
    chars_per_token: float = 4.0

    def estimate_tokens(self, text: str) -> int:
        """Rough size in tokens of a chunk of prompt text.

        Char count over the running chars/token estimate rather than a fixed
        divisor: the ratio is calibrated from the calls this run has already
        made, so a LaTeX-dense prompt on a dense tokenizer is sized as it
        actually is, and the estimate converges on the loaded model's truth.
        Floored at one token: a prompt, however short, is not free.
        """
        return max(1, int(len(str(text)) / max(_RUNNING_CHARS_PER_TOKEN, 1.0)))


# There is deliberately no per-model sampling table here. The sampler values
# a model's card recommends (top_k, top_p, ...) cannot be probed off the
# server, but rather than hardcode a handful we let a model run on its own
# card and defaults: profile.sampling above stays empty by default, and the
# per-call options in main.py layer on top of it. num_predict is the same idea
# — a per-call budget in main.py, sized by the workload, not a model property.


# ---------------------------------------------------------------------------
# Errors the server's own 400s make
# ---------------------------------------------------------------------------

class ContextLengthError(RuntimeError):
    """The server rejected a call: the prompt plus the requested generation
    outruns its context window.

    server_limit is the server's own number, parsed from the body of its
    400 (OpenAI- and vLLM-style); prompt_tokens is the prompt size the
    server reported, when it gave one. This is a configuration fact, not a
    transport hiccup: within a run the prompt only grows, so a call of this
    shape cannot succeed on retry. It is deliberately not a
    requests.RequestException, so the "transport error" retry loops in
    main.reason / main.extract do not swallow it and burn three attempts
    on a 400 that is guaranteed to repeat."""

    def __init__(self, server_limit: int,
                 prompt_tokens: Optional[int] = None) -> None:
        self.server_limit = server_limit
        self.prompt_tokens = prompt_tokens
        super().__init__(
            f"server rejected the call: its context window is "
            f"{server_limit} tokens"
            + (f" but the prompt alone is {prompt_tokens} tokens"
               if prompt_tokens is not None else "")
        )


# OpenAI- and vLLM-style context-length 400. The body names the server's
# real window; a client that budgets above it gets this instead of a
# truncated reply, and the only fix is to budget to the server.
def _context_limit_from_error(text: str) -> Tuple[Optional[int], Optional[int]]:
    """(server_limit, prompt_tokens) out of a 400 body; (None, None) when
    the body is not a context-length error."""
    m = re.search(
        r"maximum context length is (\d+) tokens?", text, re.IGNORECASE)
    if m is None:
        return None, None
    p = re.search(
        r"your prompt contains (\d+) input tokens", text, re.IGNORECASE)
    return int(m.group(1)), (int(p.group(1)) if p is not None else None)


# ---------------------------------------------------------------------------
# The model name the server reports (Profile.model_name)
# ---------------------------------------------------------------------------

def _model_name_from_props(body: Dict[str, Any]) -> Optional[str]:
    """The file name of the model a llama-server has loaded, from /props.

    llama.cpp reports the loaded model's path (model_path); the file name of
    that is what gets recorded — not the alias it was launched with, which
    is what --model asks for. model_name / model_alias cover builds that
    report it under those keys instead."""
    path = body.get("model_path")
    if isinstance(path, str) and path:
        name = os.path.basename(path)
        if name:
            return name
    for key in ("model_name", "model_alias"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """One native tool call from a reply: its id (echoed back on the
    result), the tool's name, and the arguments as the model wrote them —
    a JSON string, parsed by the ToolSet that runs it."""
    id: str
    name: str
    arguments: str


@dataclass
class Reply:
    content: str
    thinking: str = ""
    truncated: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[ToolCall] = field(default_factory=list)
    # The reply's own "model" field: on a hosted API, the model that
    # actually answered (often with a version suffix the request did not
    # carry). A llama-server puts its --alias here, whatever was requested.
    model: str = ""


# The shapes a refusal of the `tools` field itself takes: llama-server's
# "tools param requires --jinja flag", and the hosted APIs' "does not
# support tools", "tools are not supported", "Unrecognized request argument
# supplied: tools", "Extra inputs are not permitted ... tools". An error that
# merely mentions a tool (tool_call_id, a role "tool" message) does not match.
_TOOLS_UNSUPPORTED_RE = re.compile(
    r"jinja"
    r"|(?:does not|doesn't|do not|don't|not) support\w*\W+(?:\w+\W+){0,3}?"
    r"(?:tools|tool[ _-]?(?:use|call(?:ing|s)?)|function[ _-]?call(?:ing|s)?)"
    r"|\b(?:tools|tool[ _-]?(?:use|call(?:ing|s)?)|function[ _-]?call(?:ing|s)?)"
    r"\b[^.\n]{0,40}?\b(?:not supported|unsupported|not enabled|not available)"
    r"|(?:unrecognized|unknown|unexpected|extra)\W+(?:\w+\W+){0,4}?['\"]?tools\b",
    re.IGNORECASE,
)


class ToolsUnsupported(RuntimeError):
    """The server refused a call because it carried `tools`: a llama-server
    launched without --jinja, or an endpoint without tool support. The
    caller switches to the JSON tool protocol (tools.py) and re-sends."""


class OpenAICompatibleBackend:
    # Option names main.py's option dicts use -> request fields, for a
    # llama.cpp server: the standard OpenAI fields *plus* its sampler
    # extensions (top_k, min_p, repeat_penalty), all accepted on /v1.
    _OPTION_MAP = {
        "num_predict": "max_tokens",
        "temperature": "temperature",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "top_p": "top_p",
        "top_k": "top_k",           # llama.cpp extension, accepted on /v1
        "min_p": "min_p",           # ditto
        "repeat_penalty": "repeat_penalty",
        "seed": "seed",
        # num_ctx has no request-level equivalent: it is llama-server's -c/--ctx-size
        # flag, fixed at launch. Dropped here and validated in probe() instead.
    }

    # The standard OpenAI fields only, for a hosted API. The llama.cpp
    # extensions are dropped on purpose: a strict provider (OpenAI included)
    # answers 400 to an unrecognised sampling field, so top_k/min_p there
    # would break the call. Only keys in the chosen map are copied out of
    # `merged`, so the merged options are trimmed to what this endpoint
    # accepts, with no per-field check.
    _STD_OPTION_MAP = {
        "num_predict": "max_tokens",
        "temperature": "temperature",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "top_p": "top_p",
        "seed": "seed",
    }

    def __init__(self, model: str, host: str = "http://localhost:8081",
                 timeout: int = 1800, api_key: Optional[str] = None,
                 backend: str = "auto") -> None:
        if backend not in ("auto", "llamacpp", "openai"):
            raise ValueError(f"unknown backend {backend!r}: expected one of "
                             "auto, llamacpp, openai")
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        # The dialect to speak. "auto" lets probe() decide from /props; the
        # other two override that decision for when the probe can't see the
        # real server (a proxy that eats /props in front of llama-server, a
        # provider that happens to serve one).
        self.backend = backend
        # Built once; every request — the /props and /v1/models probes and the
        # chat call itself — carries it. Empty when no key is supplied: a plain
        # local llama-server wants no Authorization header.
        self._headers = (
            {"Authorization": f"Bearer {api_key}"} if api_key else {}
        )
        self.profile: Optional[Profile] = None
        # The "model" field of the latest reply: on a hosted API, the model
        # that actually answered. See recorded_model().
        self.served_model: Optional[str] = None

    def recorded_model(self) -> str:
        """The name certificates and refutations record for the model that
        did the work.

        On a llama-server, the file name /props reports: the request's
        model field is ignored there, and the reply's model field is the
        server's --alias, which says nothing about the weights. Elsewhere,
        the model named in the latest reply, which is the model that
        answered — for a hosted API, the one --model selected, often with
        its version suffix. Before any reply, or when a server names
        nothing, the --model name is the best record there is."""
        if self.profile is not None and self.profile.model_name:
            return self.profile.model_name
        return self.served_model or self.model

    def _is_llama_cpp(self) -> bool:
        """Full llama.cpp option set + thinking knobs, or standard OpenAI?

        probe() resolves this from /props, or from the forced --backend when
        one was given. Before a probe (profile None) it falls back to the
        request: only a forced "openai" is known to be not-llama.cpp; auto and
        llamacpp keep the primary transport's behaviour, the local
        llama-server's full option set.
        """
        if self.profile is None:
            return self.backend != "openai"
        return self.profile.is_llama_cpp

    def probe(self) -> Profile:
        """Read what we can about the server before a run starts.

        Two independent checks, because they answer different questions:

        * /props is llama.cpp's. If it answers, this is a llama-server and we
          get the real context ceiling, whether the chat template thinks, and
          the file name of the model it has loaded. A hosted API has no
          /props, so its absence is not an error — it is the signature of
          "not llama.cpp", and we switch to the standard OpenAI option set and
          a default context limit.
        * /v1/models is the universal OpenAI-compatible route. We hit it when
          an API key was supplied, to make a wrong key *loud* at startup
          instead of letting every call in the run 401 (a 401/403 there is a
          definite rejection and is surfaced as auth_error; a 404 (endpoint
          absent) or a network error is not an auth failure, so it is left
          alone). Its list of models is not used for naming: on a hosted
          API it is the provider's whole catalogue, in no useful order.

        The probe's measurements (context ceiling, thinking support) are kept
        either way — they describe the server that is actually there. What
        the probe decides is the dialect, and only when --backend is auto:
        an explicit backend wins over the probe, and a mismatch is flagged in
        describe() rather than silently obeyed.
        """
        global _RUNNING_CHARS_PER_TOKEN
        prof = Profile(name=self.model, backend=self.backend)

        try:
            r = requests.get(f"{self.host}/props", timeout=30,
                             headers=self._headers)
            r.raise_for_status()
            body = r.json()
        except (requests.RequestException, ValueError):
            body = None

        if body is not None:
            prof.probed_llama_cpp = True
            gen = body.get("default_generation_settings") or {}
            n_ctx = gen.get("n_ctx") or body.get("n_ctx")
            if isinstance(n_ctx, int) and n_ctx > 0:
                prof.context_limit = n_ctx

            # Whether the loaded template has a reasoning section. llama.cpp
            # reports the chat template; qwen3.5 thinks by default. qwen3.8 keys
            # its thinking on the OpenAI-style reasoning_effort knob instead
            # (launched with --chat-template-kwargs
            # '{"reasoning_effort":"xhigh"}'), so that name marks it too.
            template = body.get("chat_template", "") or ""
            prof.supports_thinking = (
                "enable_thinking" in template
                or "</thinking>" in template
                or "reasoning_effort" in template
            )
            # The file name of the model the server has loaded
            # (Profile.model_name).
            prof.model_name = _model_name_from_props(body)
            prof.probe_ok = True

        # The /v1/models call verifies a supplied key. A 401/403 with no
        # key at all is expected (an unauthenticated call to a hosted API),
        # not an auth failure, so the call is key-gated.
        if self.api_key:
            try:
                rm = requests.get(f"{self.host}/v1/models", timeout=30,
                                  headers=self._headers)
            except requests.RequestException:
                rm = None
            if rm is not None:
                if rm.status_code in (401, 403) and self.api_key:
                    prof.auth_error = (
                        f"API key rejected by {self.host}/v1/models "
                        f"(HTTP {rm.status_code}). Check --api-key / "
                        f"$LLM_API_KEY and that it matches {self.host}."
                    )
                elif rm.status_code == 200:
                    prof.key_verified = True

        # Reconcile the dialect: auto takes whatever the probe found, the
        # other backends are taken on faith. describe() flags a mismatch.
        prof.is_llama_cpp = (
            prof.probed_llama_cpp if self.backend == "auto"
            else self.backend == "llamacpp"
        )

        # Seed the running chars/token estimate from this model's calibrated
        # value; note_usage() revises it as real usage arrives this run.
        _RUNNING_CHARS_PER_TOKEN = prof.chars_per_token

        self.profile = prof
        return prof

    def chat(self, messages: List[Dict[str, Any]], *, think: Any = None,
             schema: Optional[Dict[str, Any]] = None,
             options: Optional[Dict[str, Any]] = None,
             tools: Optional[List[Dict[str, Any]]] = None,
             tool_choice: Optional[str] = None) -> Reply:
        """One completion. tools is the OpenAI-style `tools` field (a
        ToolSet's specs()), and tool_choice its companion ("none" makes the
        model answer rather than call, with the tools still declared for the
        tool messages already in the conversation). A server that refuses
        tools raises ToolsUnsupported."""
        merged = dict(self.profile.sampling if self.profile else {})
        merged.update(options or {})

        payload: Dict[str, Any] = {"model": self.model, "messages": messages,
                                   "stream": False}
        # The full llama.cpp map (with top_k/min_p/repeat_penalty) or the
        # standard OpenAI map, depending on what probe() found. Only keys in
        # the chosen map are copied, so the merged options are trimmed to
        # what this endpoint accepts.
        option_map = (self._OPTION_MAP if self._is_llama_cpp()
                      else self._STD_OPTION_MAP)
        for src, dst in option_map.items():
            if src in merged:
                payload[dst] = merged[src]

        if self._is_llama_cpp():
            if schema is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "strict": True,
                                    "schema": schema},
                }
                # Thinking suppresses *the grammar*: with reasoning on, schema
                # enforcement is not applied at all, so the model is free to
                # emit fenced JSON — which then fails the server's own parser
                # with a 500 (ggml-org/llama.cpp#20345). Turning reasoning off
                # for this call is safe: the extraction stage does no new
                # mathematics, it only copies.
                payload["reasoning_effort"] = "none"
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            elif not think:
                payload["reasoning_effort"] = "none"
            # think truthy + no schema: omit everything, take the template
            # default.
        else:
            # A hosted API: send the standard json_schema format and nothing
            # llama.cpp-specific. reasoning_effort / chat_template_kwargs are
            # dropped because a strict provider 400s on them (and only some
            # models take reasoning_effort at all). If the model still wraps
            # its JSON in a fence, extract()'s repair pipeline in main.py
            # reclaims it — the same fallback the llama.cpp path already has.
            if schema is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "strict": True,
                                    "schema": schema},
                }

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        r = requests.post(f"{self.host}/v1/chat/completions", json=payload,
                          timeout=self.timeout, headers=self._headers)
        if r.status_code >= 400:
            limit, prompt_tokens = _context_limit_from_error(r.text)
            if limit is not None:
                raise ContextLengthError(limit, prompt_tokens)
            # llama-server without --jinja answers a tools call with an
            # error naming the flag; other endpoints say the field is not
            # supported. Only that counts: an error that merely mentions
            # tools (a bad tool_call_id, a malformed tool message) is about
            # this conversation, not about the server, and switching the
            # whole run to the JSON protocol over it would be wrong.
            if tools and _TOOLS_UNSUPPORTED_RE.search(r.text):
                raise ToolsUnsupported(
                    f"HTTP {r.status_code}: {r.text[:300]}"
                )
        r.raise_for_status()
        body = r.json()
        served = body.get("model")
        if isinstance(served, str) and served:
            self.served_model = served
        # Fold this call's real usage into the running chars/token estimate:
        # the tokens the server counted for the prompt we just sent, against
        # its char count. Every caller benefits; none has to remember to.
        note_usage(body.get("usage") or {},
                   sum(len(str(m.get("content") or "")) for m in messages))
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls: List[ToolCall] = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = (tc or {}).get("function") or {}
            if fn.get("name"):
                args = fn.get("arguments")
                calls.append(ToolCall(
                    id=str(tc.get("id") or f"call_{i}"),
                    name=str(fn["name"]),
                    arguments=(args if isinstance(args, str)
                               else json.dumps(args or {})),
                ))
        return Reply(
            tool_calls=calls,
            content=msg.get("content") or "",
            # Which key holds the trace depends on the server: vLLM used
            # `reasoning_content` up through v0.15, renamed it to `reasoning`
            # in v0.16 and dropped the old key from responses, so read both.
            # In the overlap window (v0.11.1+) the two carry the same value.
            # Either way the field is only populated when the server runs
            # with --reasoning-parser (e.g. `qwen3`); without one the model's
            # think tags stay inline in content and _split_inline_thinking
            # (main.py) splits them out.
            thinking=msg.get("reasoning_content") or msg.get("reasoning") or "",
            truncated=choice.get("finish_reason") == "length",
            usage=body.get("usage") or {},
            model=str(body.get("model") or ""),
        )


# 8080 is llama-server's own default, but it is also open-webui's, so the
# project defaults to 8081: launch llama-server with --port 8081 to match,
# or point $LLAMA_HOST / --host at wherever it listens. For a hosted API,
# pass its base URL as --host (e.g. https://api.openai.com, without the /v1
# suffix) — it does not use this default at all.
# The probe makes a wrong guess loud rather than silent: a service with no
# /props route (open-webui, say) 404s, and you get
# default_generation_settings=None instead of a confusing half-working
# session against the wrong service.
_DEFAULT_HOST = os.environ.get("LLAMA_HOST", "http://localhost:8081")
# Where an API key comes from when --api-key is not given. One name on
# purpose: the key belongs to whatever --host points at, so a per-provider env
# var (OPENAI_API_KEY, ...) would be a guess about the host. Set LLM_API_KEY
# for the host you pass.
_DEFAULT_API_KEY_ENV = "LLM_API_KEY"


def make_backend(model: str, host: Optional[str] = None,
                 timeout: int = 1800,
                 api_key: Optional[str] = None,
                 backend: str = "auto") -> OpenAICompatibleBackend:
    if not api_key:
        api_key = os.environ.get(_DEFAULT_API_KEY_ENV)
    return OpenAICompatibleBackend(model, host or _DEFAULT_HOST, timeout,
                                   api_key, backend)


def note_usage(usage: Dict[str, Any], prompt_chars: int) -> None:
    """Fold one call's token usage into the running chars/token estimate.

    The estimate starts at the calibrated chars_per_token and is only revised
    as real usage arrives, so a couple of large calls early in a run is enough
    for the compaction budget to be realistic for the rest of it.
    """
    prompt_tokens = usage.get("prompt_tokens") or 0
    if prompt_tokens and prompt_chars:
        ratio = prompt_chars / prompt_tokens
        if 0.5 <= ratio <= 8.0:     # reject anything that smells like a bug
            est = prompt_chars / prompt_tokens
            global _RUNNING_CHARS_PER_TOKEN
            # exponential moving average toward the fresh measurement
            _RUNNING_CHARS_PER_TOKEN = (0.8 * _RUNNING_CHARS_PER_TOKEN
                                        + 0.2 * est)


# A mutable module-level estimate, seeded at probe time from the profile's
# calibrated value and revised by note_usage() from each call's real usage.
# Kept out of the class on purpose: it is shared across the process and is
# the *running* figure, not a per-model constant.
_RUNNING_CHARS_PER_TOKEN: float = 4.0


def chars_per_token() -> float:
    """The running chars/token estimate: the profile's calibrated seed,
    revised by note_usage() as real usage arrives. main.py's char-based
    budgets read it here instead of the stale per-model seed."""
    return _RUNNING_CHARS_PER_TOKEN


def describe(prof: Profile, requested_ctx: int) -> str:
    """Startup banner, plus warnings when the request is unsatisfiable."""
    lines = [
        f"model={prof.name}"
        + (f" loaded={prof.model_name}" if prof.model_name else "")
        + f" ctx_limit={prof.context_limit} thinking={prof.supports_thinking}"
    ]
    if prof.auth_error:
        lines.insert(0, f"  {prof.auth_error}")
    if prof.key_verified:
        lines.append("  API key verified against /v1/models.")
    if prof.is_llama_cpp and not prof.probed_llama_cpp:
        # Forced llama.cpp against a server that has no /props route.
        lines.append(
            "  --backend llamacpp was forced, but the server has no /props "
            "route: the full llama.cpp option set (top_k/min_p) and thinking "
            "suppression are sent anyway. If this is not really a "
            "llama-server, a strict provider will 400 them — and the context "
            "limit below is a default, not a measurement."
        )
    elif prof.probed_llama_cpp and not prof.is_llama_cpp:
        # Forced standard OpenAI against a server that does answer /props.
        lines.append(
            "  --backend openai was forced, but the server answers /props "
            "(a llama.cpp server): only the standard OpenAI fields are sent "
            "and thinking is not suppressed for schema calls, so on a real "
            "llama-server those replies may come back fenced."
        )
    elif not prof.is_llama_cpp:
        lines.append(
            "  Not a llama.cpp server (no /props route): the context "
            "limit is a default unless --num-ctx is given, and only the "
            "standard OpenAI sampling fields are sent (llama.cpp's "
            "top_k/min_p are dropped)."
        )
    if requested_ctx > prof.context_limit:
        if prof.probed_llama_cpp:
            lines.append(
                f"  --num-ctx {requested_ctx} exceeds the server's "
                f"{prof.context_limit}. llama-server fixes this at launch: "
                f"restart it with -c {requested_ctx}. Clamping to "
                f"{prof.context_limit} for now."
            )
        else:
            # A default, not a measurement — and main() lets an explicit
            # --num-ctx replace it there instead of clamping to a guess. If
            # the provider's real window is smaller than the budget, the
            # first call that outruns it is 400'd and chat() turns that
            # into a loud stop naming the server's own number
            # (ContextLengthError).
            lines.append(
                f"  --num-ctx {requested_ctx} replaces the "
                f"{prof.context_limit} default on record (this endpoint "
                f"reports no context limit). If the provider's real window "
                f"is smaller, the first call that outruns it is 400'd and "
                f"the run stops with the server's own number and the "
                f"--num-ctx to re-run with."
            )
    if not prof.probe_ok and not prof.key_verified:
        lines.append(
            "  Capability probe failed and no API key was verified — is "
            "the server up, and is this the right host? Everything below is a "
            "default, not a measurement. "
            "(open-webui answers on 8080 but has no /props route.)"
        )
    if not prof.supports_thinking:
        lines.append(
            "  No thinking capability reported. The generation budget "
            "assumes a reasoning trace, so it will be far larger than needed "
            "— harmless, since num_predict is a ceiling, not a target."
        )
    return "\n".join(lines)
