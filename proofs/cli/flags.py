"""Flags shared by more than one command: the model and transport flags
of every command that calls a model, and the flags of the commands whose
agents are shown the proved lemmas."""

from __future__ import annotations

import argparse

from ..workspace.settings import (
    DEFAULT_LEMMAS_ALL, DEFAULT_LEMMAS_COMMON, DEFAULT_MY_LEMMAS,
    DEFAULT_NUM_CTX, DEFAULT_TOOL_MODE,
)


def _window_size(text: str) -> int:
    """A lemma-window size: a count, or -1 for no limit."""
    n = int(text)
    if n < -1:
        raise argparse.ArgumentTypeError(f"{n} is neither a count nor -1")
    return n


def add_backend_flags(parser: argparse.ArgumentParser) -> None:
    """--host, --api-key, --model, --backend and --num-ctx: which model the
    command talks to, over which dialect, in how large a window. Taken by
    proofs run, parse, verify and repair."""
    parser.add_argument(
        "--host", required=True,
        help="OpenAI-compatible base URL: a local llama-server, or a hosted "
             "API's base (e.g. https://api.openai.com, without the /v1 "
             "suffix — it is added automatically). Required.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key sent as 'Authorization: Bearer <key>'. Needed for a "
             "hosted API; with none given, no key is sent. Leave unset for a "
             "plain local llama-server (or one launched without --api-key).",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name: the --alias llama-server was launched with, or the "
             "provider's model id (e.g. gpt-4o). Required.",
    )
    parser.add_argument(
        "--backend", choices=("auto", "llamacpp", "openai"), default="auto",
        help=(
            "Which dialect of the OpenAI-compatible endpoint to speak. "
            "auto (default): probe /props — an answer means llama.cpp (full "
            "option set, thinking suppressed for schema calls), no answer "
            "means a generic endpoint (standard OpenAI fields). llamacpp / "
            "openai force one side of that decision, for when the probe "
            "can't see the real server (a proxy in front of llama-server, a "
            "provider that happens to serve /props)."
        ),
    )
    parser.add_argument(
        "--num-ctx", type=int, default=None,
        help=(
            "Context window in tokens. Defaults to the server's context — "
            "the -c llama-server was launched with, as probed (lower it if "
            f"VRAM is tight) — or {DEFAULT_NUM_CTX} on a hosted --host API, "
            "where nothing is probed. An explicit value replaces that "
            "default; against a probed llama-server it is clamped to the "
            "ceiling."
        ),
    )


def add_tool_flags(parser: argparse.ArgumentParser) -> None:
    """--tool-mode: taken by proofs run and proofs repair, beside the
    lemma window's flags."""
    parser.add_argument(
        "--tool-mode", choices=("auto", "json"), default=DEFAULT_TOOL_MODE,
        help=(
            "How agents call tools such as lookup_lemmas. auto (default): "
            "the server's native tool calling, falling back to a JSON "
            "protocol if the server refuses it (llama-server needs --jinja "
            "for native calls). json: the JSON protocol from the start."
        ),
    )


def add_lemma_window_flags(parser: argparse.ArgumentParser) -> None:
    """--lemmas-all and --my-lemmas, the lemma window's two sizes: taken by
    proofs run and by proofs repair, the two commands whose agents are
    shown the proved lemmas."""
    parser.add_argument(
        "--lemmas-all", type=_window_size, default=DEFAULT_LEMMAS_ALL, metavar="N",
        help=(
            "Show the agents the N most recently proved lemmas, anyone's, "
            f"in full (default: {DEFAULT_LEMMAS_ALL}; -1 for all). The rest are "
            "listed by id."
        ),
    )
    parser.add_argument(
        "--lemmas-common", type=_window_size, default=DEFAULT_LEMMAS_COMMON,
        metavar="K",
        help=(
            "Also show the K lemmas the shown lemmas cite most often, "
            f"statement only (default: {DEFAULT_LEMMAS_COMMON}; -1 for every cited "
            "lemma). Ties are broken randomly, fixed for the run."
        ),
    )
    parser.add_argument(
        "--my-lemmas", type=_window_size, default=DEFAULT_MY_LEMMAS, metavar="N",
        help=(
            "Also show the N most recently proved of your own lemmas in "
            f"full (default: {DEFAULT_MY_LEMMAS}; -1 for all). Overlap with "
            "--lemmas-all is shown once."
        ),
    )
