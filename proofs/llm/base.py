"""The interface every model server is spoken to through, and the types
that cross it.

The loops (loops/reasoning.py) never import requests, never build a
payload, and never know which model is loaded. They hold a Backend and call
a small surface:

    backend.probe() -> Profile            what the server is, read at startup
    backend.chat(messages, think=..., schema=..., options={...},
                 tools=..., tool_choice=...) -> Reply
    backend.recorded_model() -> str       the model certificates record
    backend.recorded_model_known()        whether that is read off the
                                          server yet, or only the request
    backend.estimate_tokens(text)         a prompt's size, from the running
    backend.chars_per_token()             chars/token estimate chat() keeps

Everything that differs between servers — which endpoint, which fields a
request may carry, how thinking is switched off, how a server says a
prompt is too long or that it takes no tools — lives behind it, in the
implementation (one, for now: openai_compat.py). The errors a caller acts
on are the ones here: ContextLengthError (a mis-sized window: stop, don't
retry) and ToolsUnsupported (switch to the JSON tool protocol).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    # Calibration seed, not capability: how many characters of prompt text
    # one token of context roughly buys. probe() seeds the backend's running
    # estimate with it (Backend.chars_per_token), which note_usage() revises
    # from each call's real usage; the char-based budgets read that running
    # value, not this seed.
    chars_per_token: float = 4.0


# There is deliberately no per-model sampling table here. The sampler values
# a model's card recommends (top_k, top_p, ...) cannot be probed off the
# server, but rather than hardcode a handful we let a model run on its own
# card and defaults, and only the per-call options the loops build
# (Context.reasoning_options, Context.extract_options) are sent on top of the
# server's own. num_predict is the same idea — a per-call budget sized by
# the workload (loops/reasoning.py), not a model property.


class ContextLengthError(RuntimeError):
    """The server rejected a call: the prompt plus the requested generation
    outruns its context window.

    server_limit is the server's own number, parsed from the body of its
    400 (OpenAI- and vLLM-style); prompt_tokens is the prompt size the
    server reported, when it gave one. This is a configuration fact, not a
    transport hiccup: within a run the prompt only grows, so a call of this
    shape cannot succeed on retry. It is deliberately not a
    requests.RequestException, so the "transport error" retry loops in
    reason() / extract() do not swallow it and burn three attempts
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


@dataclass
class ToolCall:
    """One native tool call from a reply: its id (echoed back on the
    result), the tool's name, and the arguments as the model wrote them —
    a JSON string, parsed by the ToolSet that runs it."""
    id: str
    name: str
    arguments: str


# The keys a reply's reasoning trace can come under, in the order chat()
# reads them (see the note there).
REASONING_KEYS = ("reasoning_content", "reasoning")


def reasoning_chars(message: Dict[str, Any]) -> int:
    """The length of the reasoning trace a message carries back to the
    server (reason() sends it with a tool-call turn when the model needs
    it): prompt text, so a prompt-size count must include it. 0 for every
    other message."""
    return sum(len(str(message[k])) for k in REASONING_KEYS if message.get(k))


@dataclass
class Reply:
    content: str
    thinking: str = ""
    truncated: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[ToolCall] = field(default_factory=list)
    # The trace under the key the server sent it ({} when there was none),
    # for a tool-call turn to carry back to a server that needs to see it
    # again (reason()).
    reasoning_field: Dict[str, Any] = field(default_factory=dict)
    # The reply's own "model" field: on a hosted API, the model that
    # actually answered (often with a version suffix the request did not
    # carry). A llama-server puts its --alias here, whatever was requested.
    model: str = ""


class ToolsUnsupported(RuntimeError):
    """The server refused a call because it carried `tools`: a llama-server
    launched without --jinja, or an endpoint without tool support. The
    caller switches to the JSON tool protocol (tools/base.py) and re-sends."""


class Backend(abc.ABC):
    """A model server. Subclasses implement probe, chat and recorded_model;
    the running chars/token estimate is kept here, per backend: chat()
    feeds every call's real usage to note_usage()."""

    #: The model name requests carry (--model).
    model: str

    def __init__(self) -> None:
        self._chars_per_token = 4.0

    @abc.abstractmethod
    def probe(self) -> Profile:
        """Read what can be read about the server before a run starts."""

    @abc.abstractmethod
    def chat(self, messages: List[Dict[str, Any]], *, think: Any = None,
             schema: Optional[Dict[str, Any]] = None,
             options: Optional[Dict[str, Any]] = None,
             tools: Optional[List[Dict[str, Any]]] = None,
             tool_choice: Optional[str] = None) -> Reply:
        """One completion. tools is the OpenAI-style `tools` field (a
        ToolSet's specs()); a server that refuses tools raises
        ToolsUnsupported, and a prompt too long for the server's window
        ContextLengthError."""

    @abc.abstractmethod
    def recorded_model(self) -> str:
        """The name certificates and refutations record for the model that
        did the work."""

    @abc.abstractmethod
    def recorded_model_known(self) -> bool:
        """Whether recorded_model() names what the server runs, rather
        than falling back to the --model request before the server has
        said."""

    def seed_chars_per_token(self, value: float) -> None:
        """Start the running estimate from a model's calibrated value."""
        self._chars_per_token = value

    def note_usage(self, usage: Dict[str, Any], prompt_chars: int) -> None:
        """Fold one call's token usage into the running chars/token
        estimate.

        The estimate starts at the calibrated chars_per_token and is only
        revised as real usage arrives, so a couple of large calls early in
        a run is enough for the compaction budget to be realistic for the
        rest of it."""
        prompt_tokens = usage.get("prompt_tokens") or 0
        if prompt_tokens and prompt_chars:
            ratio = prompt_chars / prompt_tokens
            if 0.5 <= ratio <= 8.0:     # reject anything that smells like a bug
                # exponential moving average toward the fresh measurement
                self._chars_per_token = (0.8 * self._chars_per_token
                                         + 0.2 * ratio)

    def chars_per_token(self) -> float:
        """The running chars/token estimate: the profile's calibrated seed,
        revised by note_usage() as real usage arrives."""
        return self._chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """Rough size in tokens of a chunk of prompt text.

        Char count over the running chars/token estimate rather than a
        fixed divisor: the ratio is calibrated from the calls this run has
        already made, so a LaTeX-dense prompt on a dense tokenizer is sized
        as it actually is. Floored at one token: a prompt, however short, is
        not free."""
        return max(1, int(len(str(text)) / max(self._chars_per_token, 1.0)))
