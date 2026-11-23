"""Flags shared by more than one command: the model and transport flags
of every command that calls a model, and the flags of the commands whose
agents are shown the proved lemmas."""

from __future__ import annotations

import argparse

from typing import Iterable

from ..llm import routing
from ..workspace import settings as settings_mod
from ..workspace.settings import (
    DEFAULT_NUM_CTX, DEFAULT_TOOL_MODE, DEFAULT_WINDOW_SIZE,
)

# What each settings flag sets (workspace/settings.py), before its default.
_SETTINGS_HELP = {
    "run.max_iterations":
        "Loop iterations before the run stops, per loop in a parallel run; "
        "-1 runs until the conjecture is settled or you interrupt it. Counted "
        "from the run's checkpoint, so a resumed run goes on toward the same "
        "total",
    "run.max_proof_attempts":
        "How many prover rounds the proof loop may spend on one lemma per "
        "iteration",
    "run.max_rejected_lemmas":
        "How many of the most recently rejected lemmas (lemmas that went "
        "through every prover round unproved) the planner is shown",
    "run.max_loop_restarts":
        "Parallel mode: restarts in a row of a crashed loop before it is "
        "given up for the run",
    "run.replan_every":
        "Parallel mode: lemmas committed under a plan before the next is "
        "due (the operator's replan asks for one at any time)",
    "run.citation_recheck_base":
        "Recheck each direct lemma citation with probability N^(-d), where d "
        "is its current number of acceptances; 1 always, -1 disables",
    "tools.max_rounds":
        "Tool rounds (lookup_lemmas, search_lemmas) one agent call may "
        "spend before it must answer",
    "tools.code_rounds":
        "Tool rounds one agent call offered run_code may spend",
    "tools.max_lookup":
        "Lemmas one lookup_lemmas call may return",
    "tools.max_search":
        "Lemmas one search_lemmas call may return",
    "llm.max_retries":
        "Attempts at one model call",
    "llm.max_compaction_passes":
        "Compaction-and-resume passes one truncated call may get before its "
        "partial work is handed on",
}


def _window_size(text: str) -> int:
    """A lemma-window size: a count, or -1 for no limit."""
    n = int(text)
    if n < -1:
        raise argparse.ArgumentTypeError(f"{n} is neither a count nor -1")
    return n


def _temperature(text: str) -> float:
    """--temperature's value: a number of at least 0."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} is below 0")
    return value


def add_backend_flags(parser: argparse.ArgumentParser) -> None:
    """--host, --api-key, --model, --backend, --num-ctx, --temperature,
    --effort and --route: which models the command talks to, over which
    dialect, in how large a window.
    Taken by proofs run, parse, verify and repair; model_flags reads them
    back. The first five describe one model, the default route; the
    .proofs-local.json files can route roles to others (llm/routing.py),
    and a file that routes a default makes them optional."""
    parser.add_argument(
        "--host", default=None,
        help="The API's base URL as its docs give it, version path "
             "included; nothing is appended but the route "
             "(chat/completions): http://localhost:8080/v1 for a "
             "llama-server or vLLM, https://api.openai.com/v1, "
             "https://openrouter.ai/api/v1, "
             "https://generativelanguage.googleapis.com/v1beta/openai for "
             "Gemini. With --backend anthropic it defaults to "
             "https://api.anthropic.com. Needed with --model, unless "
             ".proofs-local.json routes a default model "
             "(llm.routes.default).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key sent as 'Authorization: Bearer <key>'. Needed for a "
             "hosted API; with none given, no key is sent. Leave unset for a "
             "plain local llama-server (or one launched without --api-key). "
             "A model in llm.models names an environment variable instead "
             "(api_key_env).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name: the --alias llama-server was launched with, or the "
             "provider's model id (e.g. gpt-4o). With --host, it serves "
             "every role .proofs-local.json does not route elsewhere.",
    )
    parser.add_argument(
        "--backend", choices=routing.BACKENDS, default=None,
        help=(
            "Which dialect of the OpenAI-compatible endpoint to speak. "
            "auto (default): probe /props — an answer means llama.cpp (full "
            "option set, thinking suppressed for schema calls), no answer "
            "means a generic endpoint (standard OpenAI fields). llamacpp / "
            "openai force one side of that decision, for when the probe "
            "can't see the real server (a proxy in front of llama-server, a "
            "provider that happens to serve /props). anthropic: the "
            "Anthropic Messages API instead (--host defaults to "
            "https://api.anthropic.com)."
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
    parser.add_argument(
        "--temperature", type=_temperature, default=None, metavar="T",
        help=(
            "Sampling temperature sent with every call to the --model "
            "model. By default none is sent and the server's own applies, "
            "which is almost always what you want; a model in llm.models "
            "takes a temperature field instead."
        ),
    )
    parser.add_argument(
        "--effort", default=None, metavar="LEVEL",
        help=(
            "Reasoning effort for every call to the --model model, where its "
            "API takes one: low, medium, high, xhigh or max with --backend "
            "anthropic (checked against what the model takes at startup); "
            "none, minimal, low, medium, high or xhigh as reasoning_effort "
            "otherwise. By default none is sent and the server's own "
            "applies; a model in llm.models takes an effort field instead."
        ),
    )
    parser.add_argument(
        "--with-fallback", action="store_true",
        help=(
            "When a call's model cannot serve it — its quota or credit used "
            "up, or no answer through the retries — send the call to the "
            "fallback model: llm.routes.fallback (or --route "
            "fallback=NAME), else the default route's. Per call: the next "
            "call tries its own model first. A verification check restarts "
            "on the fallback, so its certificate names one model. Without "
            "it, a spent quota stops the command."
        ),
    )
    parser.add_argument(
        "--route", action="append", default=[], metavar="ROLE=NAME",
        help=(
            "Route ROLE (planner, prover, verifier_1, ..., a glob over them "
            "such as verifier_*, the extract or compact stage, or the "
            "fallback of --with-fallback) to the "
            "model NAME in llm.models, for this command; repeatable. Every "
            "verifier must route to one model."
        ),
    )


def model_flags(args: argparse.Namespace) -> routing.Flags:
    """The backend flags of parsed args (add_backend_flags), as the
    routing table takes them."""
    return routing.Flags(
        host=args.host, model=args.model, api_key=args.api_key,
        backend=args.backend, num_ctx=args.num_ctx,
        temperature=args.temperature, effort=args.effort,
        routes=tuple(args.route or ()), with_fallback=args.with_fallback,
    )


def add_tool_flags(parser: argparse.ArgumentParser) -> None:
    """--tool-mode: taken by proofs run and proofs repair, beside the
    lemma window's flags."""
    parser.add_argument(
        "--tool-mode", choices=("auto", "json"), default=DEFAULT_TOOL_MODE,
        help=(
            "How agents call tools such as search_lemmas. auto (default): "
            "the server's native tool calling, falling back to a JSON "
            "protocol if the server refuses it (llama-server needs --jinja "
            "for native calls). json: the JSON protocol from the start."
        ),
    )


def _strategy(text: str):
    """--strategy's value (settings_mod.parse_strategy)."""
    try:
        return settings_mod.parse_strategy(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def add_lemma_window_flags(parser: argparse.ArgumentParser) -> None:
    """--strategy and the lemma window's sizes: taken by proofs run and by
    proofs repair, the two commands whose agents are shown the proved
    lemmas (loops/strategies.py)."""
    parser.add_argument(
        "--strategy", type=_strategy, default=settings_mod.DEFAULT_STRATEGY,
        metavar="NAME[,DIAL]",
        help=(
            "Which proved lemmas the agents are shown in full; the rest are "
            "found with the search_lemmas tool. recent (default): the "
            "--lemmas-all newest lemmas and your --my-lemmas newest, half "
            "of --window-size each by default. "
            "random,BETA: a sample weighted toward deep lemmas, BETA from 0 "
            "(uniform) to 1 (the deepest). depth,JITTER: lemmas nothing "
            "cites yet with the longest citation chains beneath them, "
            "JITTER from 0 (the deepest only) to 1 (any of them, uniformly). "
            "breadth: lemmas nothing cites yet whose ancestries overlap "
            "least. relevance: the lemmas a search finds for the agent's "
            "task (the lemma being proved; the conjecture, or the latest "
            "plan, for the planner). popular: the most cited lemmas. "
            "Every other strategy shows --window-size lemmas. The "
            "lemmas are chosen when the planner starts and, in a parallel "
            "run or a repair, when a prover starts."
        ),
    )
    parser.add_argument(
        "--window-size", type=_window_size, default=DEFAULT_WINDOW_SIZE,
        metavar="N",
        help=(
            "How many lemmas the window shows (default: "
            f"{DEFAULT_WINDOW_SIZE}; -1 for all); recent shows at most half "
            "of it, rounded down, from each of --lemmas-all and --my-lemmas."
        ),
    )
    parser.add_argument(
        "--lemmas-all", type=_window_size, default=None, metavar="N",
        help=(
            "--strategy recent: show the N most recently proved lemmas, "
            "anyone's (default: half of --window-size, rounded down; -1 "
            "for all)."
        ),
    )
    parser.add_argument(
        "--my-lemmas", type=_window_size, default=None, metavar="N",
        help=(
            "--strategy recent: also show the N most recently proved of "
            "your own lemmas (default: half of --window-size, rounded "
            "down; -1 for all). "
            "Overlap with --lemmas-all is shown once."
        ),
    )


def add_settings_flags(parser: argparse.ArgumentParser,
                       keys: Iterable[str] = settings_mod.KEYS) -> None:
    """A flag for each of keys (settings_mod.FLAGS), overriding
    .proofs-local.json for this command; settings_mod.flag_values reads
    them back. Every one but --max-rejected-lemmas takes -1, no limit."""
    for key in keys:
        parser.add_argument(
            settings_mod.FLAGS[key], dest=settings_mod.flag_dest(key),
            type=int, default=None, metavar="N",
            help=f"{_SETTINGS_HELP[key]}"
                 + ("" if key in settings_mod.BOUNDED or key in ("run.max_iterations", "run.citation_recheck_base")
                    else "; -1 for no limit")
                 + f" (default: {key} in the conjecture's or the "
                 f"workspace's {settings_mod.LOCAL_FILENAME}, "
                 f"{settings_mod.DEFAULTS[key]} built in)",
        )
