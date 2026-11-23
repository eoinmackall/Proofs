"""The OpenAI-compatible transport: the one Backend there is, and the only
place in this project that talks to a model over HTTP.

Everything model-specific has to live in exactly one of two places, in order
of preference:

  * A probe  — read off the server at startup, so it is correct for whatever
    is actually loaded, no matter what the alias is: the context ceiling
    and — on a llama-server — the file name of the model it has loaded,
    from /props. See probe(). Where there is no /props, the
    model that answered is read off each reply instead (recorded_model()).
  * Transport  — HOW to talk to the server. Real code. There is exactly one:
    the OpenAI-compatible chat/completions endpoint, under the base URL
    --host gives as is (http://localhost:8080/v1, https://api.openai.com/v1,
    https://generativelanguage.googleapis.com/v1beta/openai). llama.cpp's
    llama-server serves it locally; a hosted API (OpenAI, OpenRouter,
    DeepSeek, ...) serves it at its base URL. The same transport covers both.
    The only difference is that a hosted API wants an
    `Authorization: Bearer <key>` header, which this module sends whenever an
    API key is supplied (--api-key).

The dialect of that endpoint — which sampling fields it accepts, and whether
schema calls must have thinking suppressed — is a probe decision by default
(/props answers => llama.cpp), overridable with --backend
{auto,llamacpp,openai} for the cases the probe can't see.

The thing to avoid is a third axis of `if model.startswith("qwen3.8")`
branches scattered through loops/reasoning.py. Everything genuinely
model-specific in here is probed; there is no hand-maintained model table, so
a model you have not seen before runs on its own card and its defaults.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import (
    REASONING_KEYS, Backend, ContextLengthError, Profile, QuotaExhausted,
    Reply, ToolCall, ToolsUnsupported, reasoning_chars,
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


class OpenAICompatibleBackend(Backend):
    # Option names the loops' option dicts use -> request fields, for a
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
        # ctx_budget is never sent: it is this client's own budget, which
        # the loops spend by sizing num_predict (max_tokens) per call. The
        # server's window is its launch flag (-c, --max-model-len), read
        # by probe() as the ceiling.
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

    def __init__(self, model: str, host: str,
                 timeout: int = 1800, api_key: Optional[str] = None,
                 backend: str = "auto") -> None:
        if backend not in ("auto", "llamacpp", "openai"):
            raise ValueError(f"unknown backend {backend!r}: expected one of "
                             "auto, llamacpp, openai")
        self.model = model
        # The API's base URL, as given: chat/completions and models are
        # under it. /props is llama.cpp's own route, at the server's root,
        # which is the base without a final /v1.
        self.host = host.rstrip("/")
        self.root = (self.host[:-len("/v1")] if self.host.endswith("/v1")
                     else self.host)
        self.timeout = timeout
        self.api_key = api_key
        # The dialect to speak. "auto" lets probe() decide from /props; the
        # other two override that decision for when the probe can't see the
        # real server (a proxy that eats /props in front of llama-server, a
        # provider that happens to serve one).
        self.backend = backend
        # Built once; every request — the /props and models probes and the
        # chat call itself — carries it. Empty when no key is supplied: a plain
        # local llama-server wants no Authorization header.
        self._headers = (
            {"Authorization": f"Bearer {api_key}"} if api_key else {}
        )
        super().__init__()
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

    def recorded_model_known(self) -> bool:
        """Known once /props has named the file or a reply has named the
        model; before that, recorded_model() is only the --model request."""
        return bool((self.profile is not None and self.profile.model_name)
                    or self.served_model)

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
          get the real context ceiling and the file name of the model it
          has loaded. A hosted API has no
          /props, so its absence is not an error — it is the signature of
          "not llama.cpp", and we switch to the standard OpenAI option set and
          a default context limit.
        * models (under the base URL) is the universal OpenAI-compatible
          route, and every server answers it: it is the check that this
          base URL serves this model. Anything but a 200 is fatal (error):
          no answer (is the server up?), a 404 (the base URL is wrong —
          usually its version path, /v1, is missing), a 401/403 (the key is
          rejected, or one is needed). A 200 verifies a supplied key; its
          list says whether the model is served (a warning when not: a
          provider's list is not always complete), and on vLLM
          (max_model_len) or OpenRouter (context_length) the model's entry
          gives its real context window.

        The probe's measurements (context ceiling, loaded model) are kept
        either way — they describe the server that is actually there. What
        the probe decides is the dialect, and only when --backend is auto:
        an explicit backend wins over the probe, and a mismatch is flagged in
        describe() rather than silently obeyed.
        """
        prof = Profile(name=self.model, backend=self.backend)

        try:
            r = requests.get(f"{self.root}/props", timeout=30,
                             headers=self._headers)
            r.raise_for_status()
            body = r.json()
        except (requests.RequestException, ValueError):
            body = None

        if body is not None:
            prof.probed_llama_cpp = True
            prof.context_probed = True
            gen = body.get("default_generation_settings") or {}
            n_ctx = gen.get("n_ctx") or body.get("n_ctx")
            if isinstance(n_ctx, int) and n_ctx > 0:
                prof.context_limit = n_ctx

            # The file name of the model the server has loaded
            # (Profile.model_name).
            prof.model_name = _model_name_from_props(body)
            prof.probe_ok = True

        self._probe_models(prof)

        # Reconcile the dialect: auto takes whatever the probe found, the
        # other backends are taken on faith. describe() flags a mismatch.
        prof.is_llama_cpp = (
            prof.probed_llama_cpp if self.backend == "auto"
            else self.backend == "llamacpp"
        )
        if prof.error is None:
            prof.tokenizer = self._probe_tokenizer(prof.probed_llama_cpp)

        # Seed the running chars/token estimate from this model's calibrated
        # value; note_usage() revises it as real usage arrives this run.
        self.seed_chars_per_token(prof.chars_per_token)

        self.profile = prof
        return prof

    def _probe_models(self, prof: Profile) -> None:
        """The models route (see probe()): fatal unless it answers 200."""
        url = f"{self.host}/models"
        try:
            rm = requests.get(url, timeout=30, headers=self._headers)
        except requests.RequestException as e:
            prof.error = (f"cannot reach {url} ({e}): is the server up, and "
                          f"is {self.host} its base URL?")
            return
        if rm.status_code in (401, 403):
            prof.error = (
                f"API key rejected by {url} (HTTP {rm.status_code}). Check "
                f"the key (--api-key, or the variable the model's "
                f"api_key_env names) and that it matches {self.host}."
                if self.api_key else
                f"{url} needs an API key (HTTP {rm.status_code}): pass "
                f"--api-key, or name its variable with api_key_env."
            )
            return
        if rm.status_code == 404:
            prof.error = (
                f"{url} answered 404: {self.host} is not the API's base URL. "
                f"Give it with its version path, as the server's docs do: "
                f"http://localhost:8080/v1, https://api.openai.com/v1, "
                f"https://generativelanguage.googleapis.com/v1beta/openai."
            )
            return
        if rm.status_code != 200:
            prof.error = (f"{url} answered HTTP {rm.status_code}: "
                          f"{rm.text[:200]}")
            return
        prof.probe_ok = True
        prof.key_verified = bool(self.api_key)
        try:
            entries = [e for e in (rm.json().get("data") or [])
                       if isinstance(e, dict)]
        except (ValueError, AttributeError):
            entries = []
        ids = [str(e.get("id") or "") for e in entries]
        entry = next((e for e, i in zip(entries, ids)
                      if i == self.model or i.endswith("/" + self.model)), None)
        if prof.probed_llama_cpp or not ids:
            return
        prof.model_listed = entry is not None
        prof.listed = ids[:5]
        prof.listed_count = len(ids)
        window = entry and (entry.get("max_model_len")
                            or entry.get("context_length"))
        if isinstance(window, int) and window > 0:
            prof.context_limit = window
            prof.context_probed = True

    # Token counting: the server's own tokenizer and chat template, which
    # self-hosted servers expose and hosted APIs do not. Neither route
    # runs the model — it is a CPU-side lookup, tens to hundreds of ms
    # even for a full window — so a count before every call is cheap next
    # to the call it sizes.
    #
    #   vllm      POST <root>/tokenize with the chat messages (and tools):
    #             vLLM applies the chat template and answers {"count"}.
    #   llamacpp  POST <root>/apply-template with the messages (and tools)
    #             for the templated prompt, then POST <root>/tokenize
    #             with it as content: {"tokens": [...]}.
    _COUNT_TIMEOUT = 60

    def _probe_tokenizer(self, llama_cpp: bool) -> Optional[str]:
        """Which counting route the server answers (see above), tried on a
        one-message prompt; None when neither does."""
        probe = [{"role": "user", "content": "x"}]
        order = ("llamacpp", "vllm") if llama_cpp else ("vllm", "llamacpp")
        for route in order:
            if self._count(route, probe, None) is not None:
                return route
        return None

    def _count(self, route: str, messages: List[Dict[str, Any]],
               tools: Optional[List[Dict[str, Any]]]) -> Optional[int]:
        """One count over route; None on any failure."""
        try:
            if route == "vllm":
                body: Dict[str, Any] = {"model": self.model,
                                        "messages": messages,
                                        "add_generation_prompt": True}
                if tools:
                    body["tools"] = tools
                r = requests.post(f"{self.root}/tokenize", json=body,
                                  timeout=self._COUNT_TIMEOUT,
                                  headers=self._headers)
                if r.status_code != 200:
                    return None
                count = r.json().get("count")
            else:
                body = {"messages": messages}
                if tools:
                    body["tools"] = tools
                r = requests.post(f"{self.root}/apply-template", json=body,
                                  timeout=self._COUNT_TIMEOUT,
                                  headers=self._headers)
                if r.status_code != 200:
                    return None
                prompt = r.json().get("prompt")
                if not isinstance(prompt, str):
                    return None
                r = requests.post(f"{self.root}/tokenize",
                                  json={"content": prompt},
                                  timeout=self._COUNT_TIMEOUT,
                                  headers=self._headers)
                if r.status_code != 200:
                    return None
                tokens = r.json().get("tokens")
                count = len(tokens) if isinstance(tokens, list) else None
        except (requests.RequestException, ValueError, AttributeError):
            return None
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return None
        return count

    def count_tokens(self, messages: List[Dict[str, Any]],
                     tools: Optional[List[Dict[str, Any]]] = None
                     ) -> Optional[int]:
        """The server's count of this prompt (Backend.count_tokens), over
        the route probe() found; None when it found none, or when this
        count failed (the caller estimates that one call instead)."""
        route = self.profile.tokenizer if self.profile is not None else None
        if route is None:
            return None
        count = self._count(route, messages, tools)
        if count is not None:
            self.note_usage(
                {"prompt_tokens": count},
                sum(len(str(m.get("content") or "")) + reasoning_chars(m)
                    for m in messages))
        return count

    def chat(self, messages: List[Dict[str, Any]], *, think: Any = None,
             schema: Optional[Dict[str, Any]] = None,
             options: Optional[Dict[str, Any]] = None,
             tools: Optional[List[Dict[str, Any]]] = None,
             tool_choice: Optional[str] = None) -> Reply:
        """One completion. tools is the OpenAI-style `tools` field (a
        ToolSet's specs()), and tool_choice its companion. reason() never
        sends "none": a vLLM server drops the call a model writes anyway
        and returns an empty reply, so a conversation whose tools are
        closed is sent without them instead. A server that refuses tools
        raises ToolsUnsupported."""
        merged = dict(options or {})

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
            # its JSON in a fence, extract()'s repair pipeline (loops/json_repair.py)
            # reclaims it — the same fallback the llama.cpp path already has.
            if schema is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "strict": True,
                                    "schema": schema},
                }

        # A configured effort (llm.models.NAME.effort, or --effort) as
        # reasoning_effort, on every call that has not set it already: a
        # llama.cpp schema call keeps its "none", which the grammar needs.
        if merged.get("effort") and "reasoning_effort" not in payload:
            payload["reasoning_effort"] = merged["effort"]

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        r = requests.post(f"{self.host}/chat/completions", json=payload,
                          timeout=self.timeout, headers=self._headers)
        if r.status_code >= 400:
            # A spent quota or credit: 402 (OpenRouter and others), or
            # OpenAI's 429 insufficient_quota. Any other 429 is a rate
            # limit, which passes, and goes to the loops' retries.
            if r.status_code == 402 or (r.status_code == 429
                                        and "insufficient_quota" in r.text):
                raise QuotaExhausted(f"HTTP {r.status_code}: {r.text[:300]}")
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
        self.note_usage(body.get("usage") or {},
                   sum(len(str(m.get("content") or "")) + reasoning_chars(m)
                       for m in messages))
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
        reasoning_key = next((k for k in REASONING_KEYS if msg.get(k)), None)
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
            # (loops/reasoning.py) splits them out.
            thinking=msg.get("reasoning_content") or msg.get("reasoning") or "",
            truncated=choice.get("finish_reason") == "length",
            usage=body.get("usage") or {},
            model=str(body.get("model") or ""),
            reasoning_field=({reasoning_key: msg[reasoning_key]}
                             if reasoning_key else {}),
        )
