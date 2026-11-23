"""The Anthropic Messages API transport: a Backend (llm/base.py) for Claude,
spoken over plain HTTP with requests, as openai_compat.py speaks its own.

Selected with backend "anthropic" (a model's llm.models entry, or
--backend anthropic); the host defaults to https://api.anthropic.com, the
base URL Anthropic documents. The messages and models routes are under
<host>/v1; a host already ending in /v1 is taken as that base itself. The
loops build OpenAI-style messages and options for every backend; this
module translates them, and back:

  * the system message becomes the top-level `system`, one text block with
    a cache breakpoint: an agent's system prompt is the same on every call
    of its role, so a repeated role reads it from the cache. A call that
    offers tools — a conversation that grows round by round — also gets
    the automatic breakpoint on its tail; a one-shot call does not, since
    there it would only pay the write premium on bytes never read back.
  * an assistant turn with tool calls becomes text and tool_use blocks, or,
    when the reply carried thinking, its original content verbatim
    (ANTHROPIC_CONTENT_KEY): thinking blocks go back unchanged with their
    tool calls, as the API requires. echoes_reasoning makes the loops send
    them from the first tool round.
  * tool results become tool_result blocks, consecutive ones in one user
    message.
  * a schema call (extract()) sends output_config.format with the schema
    made strict (additionalProperties false on every object), and no
    thinking request; a reasoning call (think truthy) asks for adaptive
    thinking with a summarized display, so the context-wall rescue has a
    trace to compact.

A configured effort (llm.models.NAME.effort, or --effort) goes in
output_config.effort on every call; the probe reads which levels the model
takes, and a level it does not take is refused before the first call.

No sampling field is ever sent: on API version 2026-10-21 temperature,
top_p and top_k are refused on every model (routing.py refuses a
temperature on an anthropic model before a call is made). max_tokens is
the loops' num_predict, held to the model's output cap.

The probe reads GET /v1/models/{model}: the context window
(max_input_tokens), the output cap (max_tokens) and whether the model takes
adaptive thinking and structured outputs. A rejected key or an unknown
model is reported there, before a token is spent. Where the probe could
not read them, adaptive thinking and structured outputs are assumed (every
current model takes both), and a max_tokens the API refuses as above the
cap teaches the cap and is sent again.

Requests are not streamed: one POST per call, waited for under the
command's read timeout.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import (
    ANTHROPIC_CONTENT_KEY, Backend, ContextLengthError, Profile,
    QuotaExhausted, Reply, ToolCall, reasoning_chars,
)

DEFAULT_HOST = "https://api.anthropic.com"
API_VERSION = "2026-10-21"
# The stop reasons that mean the reply was cut off, not finished.
_TRUNCATED = ("max_tokens", "model_context_window_exceeded")
# The output cap assumed before the probe has read the real one, and when
# it could not: a refusal naming the real cap replaces it (chat()).
_FALLBACK_MAX_OUTPUT = 64000

# The API's context-length refusals, two wordings: (prompt tokens, limit).
_PROMPT_TOO_LONG_RE = re.compile(
    r"prompt is too long:\s*(\d+)\s*tokens\s*>\s*(\d+)", re.IGNORECASE)
_EXCEEDS_CONTEXT_RE = re.compile(
    r"exceed context limit:\s*(\d+)\s*\+\s*\d+\s*>\s*(\d+)", re.IGNORECASE)
# A max_tokens above the model's output cap: (asked, cap).
_MAX_TOKENS_RE = re.compile(r"max_tokens:\s*(\d+)\s*>\s*(\d+)", re.IGNORECASE)


def _error_message(r: requests.Response) -> str:
    """The error's message from the API's JSON body, else the raw text."""
    try:
        body = r.json()
        message = (body.get("error") or {}).get("message")
        if isinstance(message, str) and message:
            return message
    except ValueError:
        pass
    return r.text


def strict_schema(schema: Any) -> Any:
    """schema with additionalProperties false on every object, as the
    API's structured outputs require; a schema that sets it already keeps
    its own value."""
    if isinstance(schema, list):
        return [strict_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: strict_schema(v) for k, v in schema.items()}
    if out.get("type") == "object" and "additionalProperties" not in out:
        out["additionalProperties"] = False
    return out


def _tool_specs(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI-style tool specs (a ToolSet's specs()) as the API's tools."""
    out = []
    for spec in tools:
        fn = spec.get("function") or spec
        out.append({
            "name": fn["name"],
            "description": fn.get("description") or "",
            "input_schema": fn.get("parameters") or {"type": "object",
                                                      "properties": {}},
        })
    return out


def _tool_input(arguments: Any) -> Dict[str, Any]:
    """A tool call's arguments, as the model wrote them, as tool_use input."""
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def translate(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """The loops' OpenAI-style messages as (system, messages) for the
    Messages API (see the module docstring)."""
    system: List[str] = []
    out: List[Dict[str, Any]] = []

    def add(role: str, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for m in messages:
        role = m.get("role")
        content = m.get("content")
        text = content if isinstance(content, str) else ("" if content is None
                                                          else str(content))
        if role == "system":
            if text:
                system.append(text)
        elif role == "assistant":
            original = m.get(ANTHROPIC_CONTENT_KEY)
            if original:
                add("assistant", [dict(b) for b in original])
                continue
            blocks: List[Dict[str, Any]] = []
            if text.strip():
                blocks.append({"type": "text", "text": text})
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                blocks.append({"type": "tool_use", "id": call.get("id"),
                               "name": fn.get("name"),
                               "input": _tool_input(fn.get("arguments"))})
            add("assistant", blocks)
        elif role == "tool":
            add("user", [{"type": "tool_result",
                          "tool_use_id": m.get("tool_call_id"),
                          "content": text or "(empty)"}])
        else:
            if text:
                add("user", [{"type": "text", "text": text}])
    return "\n\n".join(system), out


class AnthropicBackend(Backend):
    #: Thinking blocks go back with their tool calls (Model.echo_reasoning).
    echoes_reasoning = True
    #: No sampling fields: the API refuses them (module docstring).
    accepts_sampling = False

    def __init__(self, model: str, host: Optional[str] = None,
                 timeout: int = 1800,
                 api_key: Optional[str] = None) -> None:
        super().__init__()
        self.model = model
        self.host = (host or DEFAULT_HOST).rstrip("/")
        # The versioned API base: the documented host plus /v1, or a host
        # given with its /v1 already.
        self.api = self.host if self.host.endswith("/v1") else self.host + "/v1"
        self.timeout = timeout
        self.api_key = api_key
        self._headers = {"anthropic-version": API_VERSION,
                         "content-type": "application/json"}
        if api_key:
            self._headers["x-api-key"] = api_key
        self.profile: Optional[Profile] = None
        self.served_model: Optional[str] = None
        # What the probe read (None: not known, and assumed).
        self.max_output: Optional[int] = None
        self.adaptive_thinking: Optional[bool] = None
        self.structured_outputs: Optional[bool] = None
        # The effort levels the model takes, by level (None: not probed).
        self.efforts: Optional[Dict[str, bool]] = None

    def recorded_model(self) -> str:
        """The model the API names: the Models API's id once probed, the
        latest reply's model field, else the request."""
        if self.profile is not None and self.profile.model_name:
            return self.profile.model_name
        return self.served_model or self.model

    def recorded_model_known(self) -> bool:
        return bool((self.profile is not None and self.profile.model_name)
                    or self.served_model)

    def probe(self) -> Profile:
        """GET /v1/models/{model}: the window, the output cap and the
        capabilities, and a check of the key and the model name. Anything
        but a 200 is fatal (Profile.error): no answer, a 401/403, a 404 —
        every call would fail the same way."""
        prof = Profile(name=self.model, backend="anthropic")
        try:
            r = requests.get(f"{self.api}/models/{self.model}",
                             timeout=30, headers=self._headers)
        except requests.RequestException as e:
            prof.error = (f"cannot reach {self.api}/models ({e}): is "
                          f"{self.host} the API's base URL?")
            r = None
        if r is not None:
            if r.status_code in (401, 403):
                prof.error = (
                    f"API key rejected by {self.api}/models "
                    f"(HTTP {r.status_code}): {_error_message(r)[:200]}. Check "
                    f"the key (--api-key, or the variable the model's "
                    f"api_key_env names)."
                    if self.api_key else
                    f"{self.host} needs an API key (HTTP {r.status_code}): "
                    f"name its variable with api_key_env, or pass --api-key."
                )
            elif r.status_code == 404:
                prof.error = (
                    f"{self.host} has no model {self.model!r} (HTTP 404): "
                    f"check the model id."
                )
            elif r.status_code == 200:
                try:
                    body = r.json()
                except ValueError:
                    body = {}
                self._read_model(body, prof)
            else:
                prof.error = (f"{self.api}/models/{self.model} answered HTTP "
                              f"{r.status_code}: {_error_message(r)[:200]}")
        self.seed_chars_per_token(prof.chars_per_token)
        self.profile = prof
        return prof

    def _read_model(self, body: Dict[str, Any], prof: Profile) -> None:
        prof.probe_ok = True
        prof.key_verified = bool(self.api_key)
        if isinstance(body.get("id"), str) and body["id"]:
            prof.model_name = body["id"]
        window = body.get("max_input_tokens")
        if isinstance(window, int) and window > 0:
            prof.context_limit = window
            prof.context_probed = True
        cap = body.get("max_tokens")
        if isinstance(cap, int) and cap > 0:
            self.max_output = cap
        caps = body.get("capabilities") or {}

        def supported(*path: str) -> Optional[bool]:
            node: Any = caps
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    return None
                node = node[key]
            return bool(node.get("supported")) if isinstance(node, dict) else None

        self.adaptive_thinking = supported("thinking", "types", "adaptive")
        self.structured_outputs = supported("structured_outputs")
        if "effort" in caps:
            self.efforts = {
                level: bool(supported("effort")) and bool(supported("effort", level))
                for level in ("low", "medium", "high", "xhigh", "max")
            }

    def effort_error(self, effort: str) -> Optional[str]:
        if self.efforts is None or self.efforts.get(effort):
            return None
        taken = [level for level, ok in self.efforts.items() if ok]
        return (f"{self.model} takes no effort {effort!r} (it takes "
                f"{', '.join(taken) or 'no effort setting'})")

    def _payload(self, messages: List[Dict[str, Any]], think: Any,
                 schema: Optional[Dict[str, Any]],
                 options: Dict[str, Any],
                 tools: Optional[List[Dict[str, Any]]],
                 tool_choice: Optional[str]) -> Dict[str, Any]:
        system, converted = translate(messages)
        if schema is not None and self.structured_outputs is False:
            # No structured outputs on this model: the schema moves into
            # the prompt, as extract()'s own fallback does.
            system += ("\n\nRespond with ONLY a JSON object matching this "
                       "schema, no fences, no commentary: " + json.dumps(schema))
        cap = self.max_output or _FALLBACK_MAX_OUTPUT
        asked = options.get("num_predict")
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(1, min(int(asked), cap)) if asked else cap,
            "messages": converted,
        }
        if system:
            payload["system"] = [{"type": "text", "text": system,
                                  "cache_control": {"type": "ephemeral"}}]
        if think and schema is None and self.adaptive_thinking is not False:
            payload["thinking"] = {"type": "adaptive", "display": "summarized"}
        output_config: Dict[str, Any] = {}
        if schema is not None and self.structured_outputs is not False:
            output_config["format"] = {
                "type": "json_schema", "schema": strict_schema(schema)}
        if options.get("effort"):
            output_config["effort"] = options["effort"]
        if output_config:
            payload["output_config"] = output_config
        if tools:
            payload["tools"] = _tool_specs(tools)
            if tool_choice in ("auto", "none"):
                payload["tool_choice"] = {"type": tool_choice}
            # A tool conversation grows by rounds: the automatic breakpoint
            # follows its tail, so each round reads the one before.
            payload["cache_control"] = {"type": "ephemeral"}
        return payload

    def chat(self, messages: List[Dict[str, Any]], *, think: Any = None,
             schema: Optional[Dict[str, Any]] = None,
             options: Optional[Dict[str, Any]] = None,
             tools: Optional[List[Dict[str, Any]]] = None,
             tool_choice: Optional[str] = None) -> Reply:
        """One Messages API call (see the module docstring). A prompt too
        long for the window raises ContextLengthError; any other HTTP error
        is raised as requests.HTTPError for the loops' retry and rescue."""
        payload = self._payload(messages, think, schema, dict(options or {}),
                                tools, tool_choice)
        for _attempt in range(2):
            r = requests.post(f"{self.api}/messages", json=payload,
                              timeout=self.timeout, headers=self._headers)
            if r.status_code < 400:
                break
            message = _error_message(r)
            # A spent balance or spend limit: billing_error (402), the
            # spend-cap 429 (enforced_spend_limit_reached, no retry-after),
            # or the 400 for a credit balance too low. A rate-limit 429 is
            # not one: it passes, and goes to the loops' retries.
            if (r.status_code == 402 or "billing_error" in r.text
                    or "enforced_spend_limit_reached" in r.text
                    or "credit balance is too low" in message.lower()):
                raise QuotaExhausted(f"HTTP {r.status_code}: {message[:300]}")
            m = _MAX_TOKENS_RE.search(message)
            if r.status_code == 400 and m and int(m.group(2)) < payload["max_tokens"]:
                # The output cap, learned from the refusal: sent again.
                self.max_output = int(m.group(2))
                payload["max_tokens"] = self.max_output
                continue
            m = (_PROMPT_TOO_LONG_RE.search(message)
                 or _EXCEEDS_CONTEXT_RE.search(message))
            if m:
                raise ContextLengthError(int(m.group(2)), int(m.group(1)))
            if "prompt is too long" in message.lower():
                raise ContextLengthError(
                    self.profile.context_limit if self.profile else 0)
            break
        r.raise_for_status()
        body = r.json()
        served = body.get("model")
        if isinstance(served, str) and served:
            self.served_model = served

        raw_usage = body.get("usage") or {}
        prompt_tokens = sum(int(raw_usage.get(k) or 0) for k in (
            "input_tokens", "cache_read_input_tokens",
            "cache_creation_input_tokens"))
        usage = {**raw_usage, "prompt_tokens": prompt_tokens,
                 "completion_tokens": int(raw_usage.get("output_tokens") or 0)}
        # The prompt's size counted as the loops' estimator counts it
        # (reasoning._estimate_tokens: content, tool calls, reasoning), not
        # as the JSON sent — its escapes (a doubled backslash for every
        # LaTeX command, six characters for a ∀) would inflate the ratio
        # and make every later prompt look smaller than it is.
        self.note_usage(usage, sum(
            len(str(m.get("content") or ""))
            + len(json.dumps(m.get("tool_calls") or ""))
            + reasoning_chars(m)
            for m in messages))

        blocks = body.get("content") or []
        text: List[str] = []
        thinking: List[str] = []
        calls: List[ToolCall] = []
        has_thinking = False
        for b in blocks:
            kind = b.get("type")
            if kind == "text":
                text.append(b.get("text") or "")
            elif kind == "thinking":
                has_thinking = True
                thinking.append(b.get("thinking") or "")
            elif kind == "redacted_thinking":
                has_thinking = True
            elif kind == "tool_use":
                calls.append(ToolCall(id=str(b.get("id")),
                                      name=str(b.get("name")),
                                      arguments=json.dumps(b.get("input") or {})))
        stop = body.get("stop_reason")
        return Reply(
            # A refusal's partial text is no answer: the loops see an empty
            # reply and retry, as for any other empty one.
            content="" if stop == "refusal" else "".join(text),
            thinking="\n\n".join(t for t in thinking if t),
            truncated=stop in _TRUNCATED,
            usage=usage,
            tool_calls=calls,
            # The whole content, in order, for a tool-call turn to send back
            # verbatim: its thinking blocks must go back unchanged.
            reasoning_field=({ANTHROPIC_CONTENT_KEY: blocks}
                             if has_thinking else {}),
            model=str(served or ""),
        )
