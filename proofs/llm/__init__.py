"""The connection to the model servers: one Backend (llm/base.py) per
configured model, made and probed by connect(); which model serves which
role is routing.py's.

The transport is OpenAI-compatible (openai_compat.py): a local llama-server
or a hosted API at its base URL, which is all that is needed for now; a
server speaking something else is a new Backend beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .base import (
    Backend, ContextLengthError, Profile, QuotaExhausted, Reply, ToolCall,
    ToolsUnsupported, reasoning_chars,
)
from .anthropic import AnthropicBackend
from .openai_compat import OpenAICompatibleBackend
from .routing import Flags, Model, Models, RoutingError

__all__ = [
    "AnthropicBackend", "Backend", "Connection", "ContextLengthError",
    "ProbeError",
    "Flags", "Model",
    "Models", "Profile", "QuotaExhausted", "Reply", "RoutingError", "ToolCall",
    "ToolsUnsupported", "connect", "describe", "make_backend",
    "reasoning_chars",
]


def make_backend(model: str, host: str,
                 timeout: int = 1800,
                 api_key: Optional[str] = None,
                 backend: str = "auto") -> Backend:
    # No fallbacks: the model and host are always the caller's (--model,
    # --host), and a key is sent only when one is passed (--api-key). The
    # backend picks the transport: "anthropic" the Messages API
    # (anthropic.py), every other value the OpenAI-compatible one, in its
    # dialect.
    if backend == "anthropic":
        return AnthropicBackend(model, host, timeout, api_key or None)
    return OpenAICompatibleBackend(model, host, timeout, api_key or None,
                                   backend)


class ProbeError(Exception):
    """The startup probe found the model unservable — the server
    unreachable, the base URL wrong, the key rejected or missing, the model
    unknown: every call would fail, so the command stops before it starts."""


@dataclass
class Connection:
    backend: Backend
    profile: Profile
    # The context window the command runs with (see connect()).
    num_ctx: int
    # Startup notes about the probe and the request (describe()); "" when
    # there are none.
    notes: str


def connect(model: str, host: str, timeout: int, api_key: Optional[str],
            backend: str, num_ctx: Optional[int]) -> Connection:
    """Make the backend, probe it, and settle the context window.

    The probe tells the real context ceiling. With no num_ctx (--num-ctx)
    the server's own is used as-is (on a llama-server the KV cache is
    already allocated at launch, so there is nothing to request); an
    explicit value always wins — clamped only when the ceiling was probed
    (a llama-server's -c, the Anthropic Models API's window: a fact), since
    on another endpoint the value on record is a default, not a
    measurement, and clamping an explicit request to a guess would defeat
    it. A model the probe found unservable (Profile.error) is not a
    "degrade to defaults" situation — every call would fail — so it raises
    ProbeError before a token is spent."""
    b = make_backend(model, host, timeout, api_key=api_key, backend=backend)
    prof = b.probe()
    if prof.error:
        raise ProbeError(prof.error)
    if num_ctx is None:
        window = prof.context_limit
    elif prof.context_probed:
        window = min(num_ctx, prof.context_limit)
    else:
        window = num_ctx
    # describe() warns when an *explicit* --num-ctx is clamped; in the auto
    # case the resolution above has already made the two agree.
    notes = describe(prof, num_ctx if num_ctx is not None else window)
    return Connection(b, prof, window, notes)


def describe(prof: Profile, requested_ctx: int) -> str:
    """Startup warnings about the probe and the request ("" when there is
    nothing to report)."""
    lines: List[str] = []
    if prof.key_verified:
        lines.append("API key verified against the models route.")
    if prof.model_listed is False:
        lines.append(
            f"The server's model list does not name {prof.name!r} (it lists "
            f"{', '.join(prof.listed)}{', ...' if prof.listed_count > len(prof.listed) else ''}): "
            f"if calls fail, check the model name.")
    if prof.is_llama_cpp and not prof.probed_llama_cpp:
        # Forced llama.cpp against a server that has no /props route.
        lines.append(
            "--backend llamacpp was forced, but the server has no /props. "
            "The full llama.cpp option set (top_k/min_p) and thinking "
            "suppression are sent anyway."
        )
    elif prof.probed_llama_cpp and not prof.is_llama_cpp:
        # Forced standard OpenAI against a server that does answer /props.
        lines.append(
            "--backend openai was forced, but the server answers /props: "
            "only the standard OpenAI fields are sent, and thinking is not "
            "suppressed for schema calls."
        )
    if requested_ctx > prof.context_limit:
        if prof.context_probed:
            lines.append(
                f"num_ctx {requested_ctx} exceeds the server's "
                f"{prof.context_limit}. Clamping to {prof.context_limit} for now."
            )
        else:
            # A default, not a measurement — and connect() lets an explicit
            # --num-ctx replace it there instead of clamping to a guess. If
            # the provider's real window is smaller than the budget, the
            # first call that outruns it is 400'd and chat() turns that
            # into a loud stop naming the server's own number
            # (ContextLengthError).
            lines.append(
                f"num_ctx {requested_ctx} replaces the "
                f"{prof.context_limit} default on record (this endpoint "
                f"reports no context limit). "
            )
    return "\n".join(lines)
