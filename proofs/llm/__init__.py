"""The connection to the model server: one Backend (llm/base.py) per
command, made and probed by connect().

The transport is OpenAI-compatible (openai_compat.py): a local llama-server
or a hosted API at its base URL, which is all that is needed for now; a
server speaking something else is a new Backend beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .base import (
    Backend, ContextLengthError, Profile, Reply, ToolCall, ToolsUnsupported,
    reasoning_chars,
)
from .openai_compat import OpenAICompatibleBackend

__all__ = [
    "Backend", "Connection", "ContextLengthError", "Profile", "Reply",
    "ToolCall", "ToolsUnsupported", "connect", "describe", "make_backend",
    "reasoning_chars",
]


def make_backend(model: str, host: str,
                 timeout: int = 1800,
                 api_key: Optional[str] = None,
                 backend: str = "auto") -> Backend:
    # No fallbacks: the model and host are always the caller's (--model,
    # --host), and a key is sent only when one is passed (--api-key).
    return OpenAICompatibleBackend(model, host, timeout, api_key or None,
                                   backend)


class AuthError(Exception):
    """The API rejected the key it was given: every call would fail."""


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
    explicit value always wins — clamped only when the ceiling is a probed
    llama-server -c (a fact), since on another endpoint the value on record
    is a default, not a measurement, and clamping an explicit request to a
    guess would defeat it. A key the API rejected is not a "degrade to
    defaults" situation — every call would fail — so it raises AuthError
    before a token is spent."""
    b = make_backend(model, host, timeout, api_key=api_key, backend=backend)
    prof = b.probe()
    if prof.auth_error:
        raise AuthError(prof.auth_error)
    if num_ctx is None:
        window = prof.context_limit
    elif prof.probed_llama_cpp:
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
    if prof.auth_error:
        lines.append(prof.auth_error)
    if prof.key_verified:
        lines.append("API key verified against /v1/models.")
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
        if prof.probed_llama_cpp:
            lines.append(
                f"--num-ctx {requested_ctx} exceeds the server's "
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
                f"--num-ctx {requested_ctx} replaces the "
                f"{prof.context_limit} default on record (this endpoint "
                f"reports no context limit). "
            )
    if not prof.probe_ok and not prof.key_verified:
        lines.append(
            "Capability probe failed and no API key was verified — is "
            "the server up, and is this the right host? "
        )
    return "\n".join(lines)
