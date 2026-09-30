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
    is actually loaded, no matter what the alias is. See probe().
  * Transport  — HOW to talk to the server. Real code. There is exactly one:
    the OpenAI-compatible /v1/chat/completions endpoint. llama.cpp's
    llama-server serves it locally; a hosted API (OpenAI, OpenRouter,
    DeepSeek, ...) serves it at its base URL. The same transport covers both.
    The only difference is that a hosted API wants an
    `Authorization: Bearer <key>` header, which this module sends whenever an
    API key is supplied (--api_key or $LLM_API_KEY).

The dialect of that endpoint — which sampling fields it accepts, and whether
schema calls must have thinking suppressed — is a probe decision by default
(/props answers => llama.cpp), overridable with --backend
{auto,llamacpp,openai} for the cases the probe can't see.

The thing to avoid is a third axis of `if model.startswith("qwen3.8")`
branches scattered through reason()/extract(). Everything genuinely
model-specific in here is probed; there is no hand-maintained model table, so
a model you have not seen before runs on its own card and its defaults.

Usage from main.py:

    from llm_backend import make_backend
    BACKEND = make_backend(args.model, args.host, api_key=args.api_key,
                           backend=args.backend)
    PROFILE = BACKEND.probe()
    if PROFILE.auth_error:            # a supplied key the API rejected
        raise SystemExit(PROFILE.auth_error)
    reply = BACKEND.chat(messages, think=..., schema=None, options={...})
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# What we know about a model
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    """What we know about the loaded model. Probed where possible."""
    name: str
    context_limit: int = 65536     # ceiling of the server's KV cache (tokens).
                                   # --num_ctx must fit inside this.
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
# The backend
# ---------------------------------------------------------------------------

@dataclass
class Reply:
    content: str
    thinking: str = ""
    truncated: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)


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
        global _RUNNING_CHARS_PER_TOKEN

        """Read what we can about the server before a run starts.

        Two independent checks, because they answer different questions:

        * /props is llama.cpp's. If it answers, this is a llama-server and we
          get the real context ceiling and whether the chat template thinks.
          A hosted API has no /props, so its absence is not an error — it is
          the signature of "not llama.cpp", and we switch to the standard
          OpenAI option set and a default context limit.
        * /v1/models is the universal OpenAI-compatible route. When an API key
          was supplied we hit it to make a wrong key *loud* at startup instead
          of letting every call in the run 401. A 401/403 there is a definite
          rejection and is surfaced as auth_error; a 404 (endpoint absent) or a
          network error is not an auth failure, so it is left alone.

        The probe's measurements (context ceiling, thinking support) are kept
        either way — they describe the server that is actually there. What
        the probe decides is the dialect, and only when --backend is auto:
        an explicit backend wins over the probe, and a mismatch is flagged in
        describe() rather than silently obeyed.
        """
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
            prof.probe_ok = True

        if self.api_key:
            try:
                rm = requests.get(f"{self.host}/v1/models", timeout=30,
                                  headers=self._headers)
            except requests.RequestException:
                rm = None
            if rm is not None:
                if rm.status_code in (401, 403):
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

    def chat(self, messages: List[Dict[str, str]], *, think: Any = None,
             schema: Optional[Dict[str, Any]] = None,
             options: Optional[Dict[str, Any]] = None) -> Reply:
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

        r = requests.post(f"{self.host}/v1/chat/completions", json=payload,
                          timeout=self.timeout, headers=self._headers)
        r.raise_for_status()
        body = r.json()
        # Fold this call's real usage into the running chars/token estimate:
        # the tokens the server counted for the prompt we just sent, against
        # its char count. Every caller benefits; none has to remember to.
        note_usage(body.get("usage") or {},
                   sum(len(str(m.get("content") or "")) for m in messages))
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return Reply(
            content=msg.get("content") or "",
            thinking=msg.get("reasoning_content") or "",
            truncated=choice.get("finish_reason") == "length",
            usage=body.get("usage") or {},
        )


# 8080 is llama-server's own default, but it is also open-webui's, and on this
# machine open-webui has it. Launch llama-server with --port 8081 to match.
# For a hosted API, pass its base URL as --host (e.g.
# https://api.openai.com/v1) — it does not use this default at all.
# The probe makes a wrong guess loud rather than silent: open-webui has no
# /props route, so it 404s and you get default_generation_settings=None
# instead of a confusing half-working session against the wrong service.
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
        f"model={prof.name} "
        f"ctx_limit={prof.context_limit} thinking={prof.supports_thinking}"
    ]
    if prof.auth_error:
        lines.insert(0, f"  ⛔ {prof.auth_error}")
    if prof.key_verified:
        lines.append("  🔑 API key verified against /v1/models.")
    if prof.is_llama_cpp and not prof.probed_llama_cpp:
        # Forced llama.cpp against a server that has no /props route.
        lines.append(
            "  ⚠️  --backend llamacpp was forced, but the server has no /props "
            "route: the full llama.cpp option set (top_k/min_p) and thinking "
            "suppression are sent anyway. If this is not really a "
            "llama-server, a strict provider will 400 them — and the context "
            "limit below is a default, not a measurement."
        )
    elif prof.probed_llama_cpp and not prof.is_llama_cpp:
        # Forced standard OpenAI against a server that does answer /props.
        lines.append(
            "  ⚠️  --backend openai was forced, but the server answers /props "
            "(a llama.cpp server): only the standard OpenAI fields are sent "
            "and thinking is not suppressed for schema calls, so on a real "
            "llama-server those replies may come back fenced."
        )
    elif not prof.is_llama_cpp:
        lines.append(
            "  ℹ️  Not a llama.cpp server (no /props route): the context "
            "limit is a default unless --num-ctx is given, and only the "
            "standard OpenAI sampling fields are sent (llama.cpp's "
            "top_k/min_p are dropped)."
        )
    if requested_ctx > prof.context_limit:
        if prof.is_llama_cpp:
            lines.append(
                f"  ⚠️  --num-ctx {requested_ctx} exceeds the server's {prof.context_limit}. "
                f"llama-server fixes this at launch: restart it with "
                f"-c {requested_ctx}. Clamping to {prof.context_limit} for now."
            )
        else:
            lines.append(
                f"  ⚠️  --num-ctx {requested_ctx} exceeds the {prof.context_limit} "
                f"on record (a default for non-llama.cpp endpoints). "
                f"Clamping; pass a larger --num-ctx if the model allows it."
            )
    if not prof.probe_ok and not prof.key_verified:
        lines.append(
            "  ⚠️  Capability probe failed and no API key was verified — is "
            "the server up, and is this the right host? Everything below is a "
            "default, not a measurement. "
            "(open-webui answers on 8080 but has no /props route.)"
        )
    if not prof.supports_thinking:
        lines.append(
            "  ⚠️  No thinking capability reported. The generation budget "
            "assumes a reasoning trace, so it will be far larger than needed "
            "— harmless, since num_predict is a ceiling, not a target."
        )
    return "\n".join(lines)
