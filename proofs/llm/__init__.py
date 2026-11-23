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
    DEFAULT_CTX_BUDGET, Backend, ContextLengthError, Profile, QuotaExhausted, Reply, ToolCall,
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
    # The context budget the command runs with (see connect()).
    ctx_budget: int
    # Startup notes about the probe and the request (describe()); "" when
    # there are none.
    notes: str


def connect(model: str, host: str, timeout: int, api_key: Optional[str],
            backend: str, ctx_budget: Optional[int]) -> Connection:
    """Make the backend, probe it, and settle the context budget.

    The budget is the most one call — prompt plus generation — may take:
    ctx_budget (--ctx-budget, llm.models.NAME.ctx_budget) when given, else
    DEFAULT_CTX_BUDGET. The server's own window is a ceiling on it, never
    the default: where the probe read one (llama.cpp's per-slot n_ctx,
    vLLM's max_model_len, the Anthropic Models API) the budget is clamped
    to it; where it read none there is nothing to clamp to, and a budget
    the server cannot hold comes back as its context-length 400
    (ContextLengthError). A model the probe found unservable
    (Profile.error) is not a "degrade to defaults" situation — every call
    would fail — so it raises ProbeError before a token is spent."""
    b = make_backend(model, host, timeout, api_key=api_key, backend=backend)
    prof = b.probe()
    if prof.error:
        raise ProbeError(prof.error)
    requested = ctx_budget if ctx_budget is not None else DEFAULT_CTX_BUDGET
    budget = (min(requested, prof.context_limit) if prof.context_probed
              else requested)
    return Connection(b, prof, budget, describe(prof, requested))


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
    if prof.context_probed and requested_ctx > prof.context_limit:
        lines.append(
            f"The context budget {requested_ctx:,} exceeds the server's "
            f"{prof.context_limit:,}-token window; using "
            f"{prof.context_limit:,}.")
    if prof.probe_ok and prof.tokenizer is None:
        lines.append(
            "Prompt sizes are estimated from their characters: the server "
            "offers no token count.")
    return "\n".join(lines)
