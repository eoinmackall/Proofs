"""Transport + capability layer for the multi-agent theorem prover.

This module keeps two things apart:

  * Transport  — HOW to talk to the server. Real code. There is exactly one:
    llama.cpp's llama-server, on its OpenAI-compatible /v1/chat/completions
    endpoint.

  * Profile  — WHAT this particular model can do (thinking? how big a
    context?). Data, and mostly *probed at startup* rather than written
    down, so adding a model needs no code change at all.

The thing to avoid is a third axis of `if model.startswith("qwen3.8")`
branches scattered through reason()/extract(). Everything genuinely
model-specific below is either probed or lives in one small OVERRIDES dict.

Usage from main.py:

    from llm_backend import make_backend
    BACKEND = make_backend(args.model, args.host)
    PROFILE = BACKEND.probe()
    reply = BACKEND.chat(messages, think=..., schema=None, options={...})
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


# ----------------------------------------------------------------------------
# Normalised request/response types
# ----------------------------------------------------------------------------
@dataclass
class Reply:
    """One chat completion, with the server-specific shapes flattened out."""
    content: str = ""
    thinking: str = ""
    truncated: bool = False        # llama.cpp finish_reason
    prompt_tokens: int = 0
    eval_tokens: int = 0


@dataclass
class Profile:
    """What we know about the loaded model. Probed where possible."""
    name: str
    context_limit: int = 65536     # tokens the *server* will actually allow;
                                  # llama-server fixes this with -c at launch
    supports_thinking: bool = False
    probe_ok: bool = False         # False => everything below is a guess
    # Sampler values the model card recommends. REASONING_OPTIONS in main.py
    # is layered on top of these, not instead of them.
    sampling: Dict[str, Any] = field(default_factory=dict)
    # Calibrated from real prompt_eval_count values; see note_usage().
    chars_per_token: float = 4.0

    def note_usage(self, prompt_chars: int, prompt_tokens: int) -> None:
        """Replace the chars//4 guess with the tokenizer's actual ratio.

        main.py's _headroom() estimates prompt size as chars // 4. That is
        already optimistic for LaTeX-dense text (\\mathrm, \\alpha, subscripts
        tokenize badly) and it drifts further every time you change model,
        because you change tokenizer. Feeding real counts back keeps the
        headroom calculation honest across models.
        """
        if prompt_tokens > 0 and prompt_chars > 0:
            observed = prompt_chars / prompt_tokens
            self.chars_per_token = 0.7 * self.chars_per_token + 0.3 * observed

    def estimate_tokens(self, text: str) -> int:
        return int(len(text) / max(self.chars_per_token, 1.0)) + 1


# ----------------------------------------------------------------------------
# The only hand-maintained model knowledge in the project.
#
# Everything here is a *recommended sampling setting* — the one thing no
# endpoint reports. Capabilities and context sizes are probed, so they are
# deliberately absent. Match is by longest prefix of the model name.
#
# presence_penalty is the delicate one: it pushes the model off tokens it
# has already used, which is actively harmful in mathematics (the same
# symbols recur by necessity) and in the extraction stage's verbatim
# copying. The right value is model-specific, so each entry states its own
# rather than inheriting a global.
# ----------------------------------------------------------------------------
OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Mirrors the reference launch line (below), so a request cannot drift
    # from the server's own defaults.
    "qwen3.8": {"top_p": 0.95, "top_k": 20, "min_p": 0.0},
    # 0.4 was tuned against Ollama's qwen3.6 Modelfile baseline
    # (presence_penalty 1.5); on llama-server it is a real sampler flag, so
    # review it if a Qwen3.5 run regresses.
    "qwen3.5": {"presence_penalty": 0.4, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
}


def _sampling_for(model: str) -> Dict[str, Any]:
    key = max(
        (k for k in OVERRIDES if model.lower().replace("-", ".").startswith(k)),
        key=len,
        default=None,
    )
    return dict(OVERRIDES.get(key, {})) if key else {}



# ----------------------------------------------------------------------------
# llama.cpp (llama-server, OpenAI-compatible endpoint)
# ----------------------------------------------------------------------------
# Reference launch for Qwen3.8-27B. The alias is what --model must match:
#
#   llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL \
#       --alias qwen3.8-27b --host 0.0.0.0 --port 8081 --no-mmproj \
#       --parallel 1 -ngl 99 --flash-attn on \
#       --cache-type-k q8_0 --cache-type-v q8_0 \
#       -c 65536 -b 512 -ub 512 --no-context-shift -n -1 \
#       --reasoning-format deepseek \
#       --chat-template-kwargs '{"reasoning_effort":"xhigh"}' \
#       --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
#
# What the probe leans on there: --port 8081 is this module's default host;
# -c fixes the context at launch, so the banner tells you to restart the
# server, not re-request, when --num-ctx is bigger; --reasoning-format
# deepseek is what splits the trace into message.reasoning_content; the
# sampler flags are mirrored in OVERRIDES["qwen3.8"] so per-request options
# cannot drift from them.

class LlamaCppBackend:
    # Option names main.py's option dicts use -> request fields.
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

    def __init__(self, model: str, host: str = "http://localhost:8081",
                 timeout: int = 1800) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.profile: Optional[Profile] = None

    def probe(self) -> Profile:
        """Read the *server's* configuration, which the client cannot change."""
        prof = Profile(name=self.model, sampling=_sampling_for(self.model))
        try:
            r = requests.get(f"{self.host}/props", timeout=30)
            r.raise_for_status()
            body = r.json()
        except (requests.RequestException, ValueError):
            return prof

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
            or "<think>" in template
            or "reasoning_effort" in template
        )
        prof.probe_ok = True
        self.profile = prof
        return prof

    def chat(self, messages: List[Dict[str, str]], *, think: Any = None,
             schema: Optional[Dict[str, Any]] = None,
             options: Optional[Dict[str, Any]] = None) -> Reply:
        merged = dict(self.profile.sampling if self.profile else {})
        merged.update(options or {})

        payload: Dict[str, Any] = {"model": self.model, "messages": messages,
                                   "stream": False}
        for src, dst in self._OPTION_MAP.items():
            if src in merged:
                payload[dst] = merged[src]

        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "strict": True, "schema": schema},
            }
            # Thinking suppresses *the grammar*: with reasoning on, schema
            # enforcement is not applied at all, so the model is free to emit
            # fenced JSON — which then fails the server's own parser with a
            # 500 (ggml-org/llama.cpp#20345). Turning reasoning off for this
            # call is safe: the extraction stage does no new mathematics, it
            # only copies.
            payload["reasoning_effort"] = "none"
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif not think:
            payload["reasoning_effort"] = "none"
        # think truthy + no schema: omit everything, take the template default.

        r = requests.post(f"{self.host}/v1/chat/completions", json=payload,
                          timeout=self.timeout)
        r.raise_for_status()
        body = r.json()

        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content") or ""
        # --reasoning-format deepseek (the default) splits the trace out here;
        # --reasoning-format none leaves it inline in content as <think>...</think>,
        # which main.py's existing _THINK_RE path already handles.
        thinking = msg.get("reasoning_content") or ""

        usage = body.get("usage") or {}
        reply = Reply(
            content=content,
            thinking=thinking,
            truncated=choice.get("finish_reason") == "length",
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            eval_tokens=usage.get("completion_tokens", 0) or 0,
        )
        if self.profile:
            self.profile.note_usage(
                sum(len(m["content"]) for m in messages), reply.prompt_tokens
            )
        return reply


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
# 8080 is llama-server's own default, but it is also open-webui's, and on this
# machine open-webui has it. Launch llama-server with --port 8081 to match.
# The probe makes a wrong guess loud rather than silent: open-webui has no
# /props route, so it 404s and you get default_generation_settings=None
# instead of a confusing half-working session against the wrong service.
_DEFAULT_HOST = os.environ.get("LLAMA_HOST", "http://localhost:8081")


def make_backend(model: str, host: Optional[str] = None,
                 timeout: int = 1800) -> LlamaCppBackend:
    return LlamaCppBackend(model, host or _DEFAULT_HOST, timeout)


def describe(prof: Profile, requested_ctx: int) -> str:
    """One-line startup banner, and a warning when the request is unsatisfiable."""
    lines = [
        f"model={prof.name} "
        f"ctx_limit={prof.context_limit} thinking={prof.supports_thinking}"
    ]
    if requested_ctx > prof.context_limit:
        lines.append(
            f"  ⚠️  --num-ctx {requested_ctx} exceeds the server's {prof.context_limit}. "
            f"llama-server fixes this at launch: restart it with "
            f"-c {requested_ctx}. Clamping to {prof.context_limit} for now."
        )
    if not prof.probe_ok:
        lines.append(
            "  ⚠️  Capability probe failed — is the server up, and is this the "
            "right host? Everything below is a default, not a measurement. "
            "(open-webui answers on 8080 but has no /props route.)"
        )
    if not prof.supports_thinking:
        lines.append(
            "  ⚠️  No thinking capability reported. The generation budget "
            "assumes a reasoning trace, so it will be far larger than needed "
            "— harmless, since num_predict is a ceiling, not a target."
        )
    return "\n".join(lines)
