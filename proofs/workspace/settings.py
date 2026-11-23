"""The run's tunables, per machine: the run.*, tools.* and llm.* keys of
the .proofs-local.json files.

A setting is resolved key by key, the first of these that sets it winning:

    flags                                   this command only (FLAGS names
                                            each key's; cli/flags.py adds them)
    conjectures/NAME/.proofs-local.json     this conjecture on this machine
    <workspace>/.proofs-local.json          this machine
    DEFAULTS                                built in

The files are the ones the code limits live in (tools/run_code/limits.py
reads their code.* keys); both are gitignored. proofs init writes the
workspace's with every key at its default, so the machine's settings are
there to edit from the start; a conjecture's holds only the keys it
overrides, and need not exist. The keys:

    run.max_iterations        loop iterations before the run stops; -1 runs
                              until the conjecture is settled or the run is
                              interrupted. In a parallel run, per loop.
    run.max_proof_attempts    prover rounds one iteration may spend on a
                              lemma before it is given up and re-planned
    run.max_rejected_lemmas   how many of the most recently rejected lemmas
                              the planner is shown
    run.max_loop_restarts     parallel mode: restarts in a row of a crashed
                              thread before it is given up for the run
    run.replan_every          parallel mode: lemmas committed under a plan
                              before the next is due; -1 re-plans only when
                              the operator asks
    run.citation_recheck_base probability base for direct citation rechecks
                              in proofs run; 1 always, -1 disabled
    tools.max_rounds          tool rounds one agent call may spend before it
                              must answer (lookup_lemmas, search_lemmas)
    tools.code_rounds         the same, for a call offered run_code: a
                              computation is written, run, read and
                              corrected, so it needs more
    tools.max_lookup          lemmas one lookup_lemmas call may return
    tools.max_search          lemmas one search_lemmas call may return
    llm.max_retries           attempts at one model call
    llm.max_compaction_passes compaction-and-resume passes one truncated call
                              may get before its partial work is handed on

run.citation_recheck_base takes -1 to disable citation checks. Every other
key but run.max_rejected_lemmas also takes -1, for no limit (a tool
call's rounds are then bounded by the context window alone). within,
below, counts and of_limit read a limit that may be -1.

Two more keys are objects rather than integers, and route the agents'
calls to models (llm/routing.py has their format and how their layers
combine):

    llm.models                the models a run may use, by name
    llm.routes                which model each role is served by
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..llm.base import DEFAULT_CTX_BUDGET  # noqa: F401  (re-exported for the flags)
from ..tools.run_code import limits

LOCAL_FILENAME = limits.LOCAL_FILENAME

DEFAULTS: Dict[str, int] = {
    "run.max_iterations": -1,
    "run.max_proof_attempts": 3,
    "run.max_rejected_lemmas": 4,
    "run.max_loop_restarts": 3,
    "run.replan_every": 5,
    "run.citation_recheck_base": 2,
    "tools.max_rounds": 8,
    "tools.code_rounds": 8,
    "tools.max_lookup": 10,
    "tools.max_search": 5,
    "llm.max_retries": 3,
    "llm.max_compaction_passes": 2,
}
KEYS = tuple(DEFAULTS)
# The routing keys (llm/routing.py): objects, layered and checked there
# rather than parsed as integers here.
MODELS_KEY = "llm.models"
ROUTES_KEY = "llm.routes"
ROUTING_KEYS = (MODELS_KEY, ROUTES_KEY)

# The Settings field each key sets.
FIELDS: Dict[str, str] = {
    "run.max_iterations": "max_iterations",
    "run.max_proof_attempts": "max_proof_attempts",
    "run.max_rejected_lemmas": "max_rejected_lemmas",
    "run.max_loop_restarts": "max_loop_restarts",
    "run.replan_every": "replan_every",
    "run.citation_recheck_base": "citation_recheck_base",
    "tools.max_rounds": "max_tool_rounds",
    "tools.code_rounds": "code_tool_rounds",
    "tools.max_lookup": "max_lookup",
    "tools.max_search": "max_search",
    "llm.max_retries": "llm_max_retries",
    "llm.max_compaction_passes": "max_compaction_passes",
}

# The defaults of the settings that are flags only (proofs run, proofs
# repair): how the lemma window is chosen (loops/strategies.py) and its
# size (-1 is no limit) and how agents call tools. The context budget's
# default, DEFAULT_CTX_BUDGET, is the llm package's (llm/base.py).
DEFAULT_WINDOW_SIZE = 30
# The window's strategies; the two that take a dial (random's beta,
# depth's jitter), a number from 0 to 1, after a comma: random,0.5.
STRATEGIES = ("recent", "random", "depth", "breadth", "relevance", "popular")
DIALS = ("random", "depth")
DEFAULT_STRATEGY: Tuple[str, float] = ("recent", 0.0)
DEFAULT_TOOL_MODE = "auto"


@dataclass
class Settings:
    """The settings a command runs with: the keys above as fields (FIELDS
    names them), and the flag-only ones."""

    max_iterations: int = DEFAULTS["run.max_iterations"]
    max_proof_attempts: int = DEFAULTS["run.max_proof_attempts"]
    max_rejected_lemmas: int = DEFAULTS["run.max_rejected_lemmas"]
    max_loop_restarts: int = DEFAULTS["run.max_loop_restarts"]
    replan_every: int = DEFAULTS["run.replan_every"]
    citation_recheck_base: int = DEFAULTS["run.citation_recheck_base"]
    max_tool_rounds: int = DEFAULTS["tools.max_rounds"]
    code_tool_rounds: int = DEFAULTS["tools.code_rounds"]
    max_lookup: int = DEFAULTS["tools.max_lookup"]
    max_search: int = DEFAULTS["tools.max_search"]
    llm_max_retries: int = DEFAULTS["llm.max_retries"]
    max_compaction_passes: int = DEFAULTS["llm.max_compaction_passes"]
    # The lemma window (loops/strategies.py): the strategy and its dial,
    # its size, and recent's two sizes, None for half the window each.
    strategy: str = DEFAULT_STRATEGY[0]
    strategy_dial: float = DEFAULT_STRATEGY[1]
    window_size: int = DEFAULT_WINDOW_SIZE
    lemmas_all: Optional[int] = None
    my_lemmas: Optional[int] = None
    # How agents call tools: "auto" or "json" (see --tool-mode).
    tool_mode: str = DEFAULT_TOOL_MODE
    # The routing keys as each file holds them, lowest layer first, as
    # (source, value) pairs: llm/routing.py combines them. The context
    # budget is each model's own (llm.routing.Model.ctx_budget).
    model_layers: List[Tuple[str, Any]] = field(default_factory=list)
    route_layers: List[Tuple[str, Any]] = field(default_factory=list)
    # How many of a proof's verifiers run at once (proofs verify and
    # proofs repair --parallel); 1 runs them one after another.
    parallel_verifiers: int = 1

    def unlimited_iterations(self) -> bool:
        return self.max_iterations < 0


# The flag that sets each key for one command (cli/flags.py,
# add_settings_flags); its argparse dest is the name in snake case.
FLAGS: Dict[str, str] = {
    "run.max_iterations": "--max-iterations",
    "run.max_proof_attempts": "--max-proof-attempts",
    "run.max_rejected_lemmas": "--max-rejected-lemmas",
    "run.max_loop_restarts": "--max-loop-restarts",
    "run.replan_every": "--replan-every",
    "run.citation_recheck_base": "--citation-recheck-base",
    "tools.max_rounds": "--max-tool-rounds",
    "tools.code_rounds": "--max-code-rounds",
    "tools.max_lookup": "--max-lookup",
    "tools.max_search": "--max-search",
    "llm.max_retries": "--max-retries",
    "llm.max_compaction_passes": "--max-compaction-passes",
}


def parse_strategy(text: str) -> Tuple[str, float]:
    """--strategy's value: a strategy, and for random and depth a dial from
    0 to 1 after a comma (random,0.5); a missing dial is 0. ValueError for
    anything else."""
    name, _, dial = (part.strip() for part in str(text).partition(","))
    name = name.lower()
    if name not in STRATEGIES:
        raise ValueError(f"{name!r} is not a strategy; the strategies are "
                         f"{', '.join(STRATEGIES)}")
    if not dial:
        return name, 0.0
    if name not in DIALS:
        raise ValueError(f"{name} takes no dial; only {' and '.join(DIALS)} do")
    try:
        value = float(dial)
    except ValueError:
        raise ValueError(f"{name}'s dial must be a number, not {dial!r}") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}'s dial must be from 0 to 1, not {value}")
    return name, value


def flag_dest(key: str) -> str:
    return FLAGS[key][2:].replace("-", "_")


def flag_values(args: Any) -> Dict[str, Any]:
    """The settings flags of parsed args, as resolve() takes them: a key
    for each flag the command has, None for one not passed."""
    return {k: getattr(args, flag_dest(k), None) for k in KEYS}


# The smallest value each key takes, -1 (no limit) aside.
_MINIMUM: Dict[str, int] = {"run.max_rejected_lemmas": 0}
# The keys that do not take -1: the reject list is written into the
# planner's prompt and the checkpoint, so it stays bounded.
BOUNDED = frozenset({"run.max_rejected_lemmas"})


def within(count: int, limit: int) -> bool:
    """count <= limit, for a limit that may be -1 (no limit)."""
    return limit < 0 or count <= limit


def below(count: int, limit: int) -> bool:
    """count < limit, for a limit that may be -1 (no limit)."""
    return limit < 0 or count < limit


def counts(limit: int, start: int = 0) -> Iterable[int]:
    """limit integers from start, or endlessly many for -1."""
    return itertools.count(start) if limit < 0 else range(start, start + limit)


def of_limit(n: int, limit: int) -> str:
    """n against its limit, as the log shows it: "n/limit", or "n" when
    there is none."""
    return str(n) if limit < 0 else f"{n}/{limit}"


class SettingsError(Exception):
    """The file is unreadable, or a value is not usable."""


def _parse(key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{key} must be an integer, not {value!r}")
    try:
        n = int(value)
    except ValueError:
        raise ValueError(f"{key} must be an integer, not {value!r}") from None
    if n == -1 and key not in BOUNDED:
        return n
    low = _MINIMUM.get(key, 1)
    if n < low:
        extra = "" if key in BOUNDED else " (or -1, for no limit)"
        if key == "run.citation_recheck_base":
            extra = " (or -1 to disable)"
        raise ValueError(f"{key} must be at least {low}{extra}, not {n}")
    return n


def read(path: Path) -> Dict[str, Any]:
    """The settings and routing keys a .proofs-local.json holds, {} when
    there is no file. Its code.* keys are limits.py's; any other key is an
    error."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SettingsError(f"{path} is not readable: {e}") from e
    if not isinstance(data, dict):
        raise SettingsError(f"{path} must hold a JSON object")
    unknown = sorted(k for k in data
                     if k not in DEFAULTS and k not in ROUTING_KEYS
                     and not k.startswith("code."))
    if unknown:
        raise SettingsError(
            f"{path}: unknown key(s) {', '.join(unknown)}; the keys are "
            f"{' '.join(KEYS + ROUTING_KEYS)} and the code.* limits"
        )
    return {k: v for k, v in data.items()
            if k in DEFAULTS or k in ROUTING_KEYS}


def local_files(
    workspace_root: Optional[Path], conjecture_dir: Optional[Path],
) -> List[Tuple[str, Path]]:
    """The .proofs-local.json files a command reads, lowest layer first,
    each with the name messages give it: its path from the workspace root
    when it is under one."""
    paths = []
    if workspace_root is not None:
        paths.append(workspace_root / LOCAL_FILENAME)
    if conjecture_dir is not None:
        paths.append(conjecture_dir / LOCAL_FILENAME)
    files = []
    for p in paths:
        label = str(p)
        if workspace_root is not None:
            try:
                label = str(p.relative_to(workspace_root))
            except ValueError:
                pass
        files.append((label, p))
    return files


def resolve(
    workspace_root: Optional[Path],
    flags: Optional[Mapping[str, Any]] = None,
    conjecture_dir: Optional[Path] = None,
) -> Settings:
    """The settings in effect (see the module docstring). flags maps a key
    to the value a flag gave, None for a flag not passed; conjecture_dir,
    when given, adds its .proofs-local.json above the workspace's."""
    values: Dict[str, int] = {}
    layers = [("default", DEFAULTS)]
    model_layers: List[Tuple[str, Any]] = []
    route_layers: List[Tuple[str, Any]] = []
    for label, path in local_files(workspace_root, conjecture_dir):
        data = read(path)
        layers.append((label, {k: v for k, v in data.items() if k in DEFAULTS}))
        if MODELS_KEY in data:
            model_layers.append((label, data[MODELS_KEY]))
        if ROUTES_KEY in data:
            route_layers.append((label, data[ROUTES_KEY]))
    if flags:
        layers.append(("flag", {k: v for k, v in flags.items() if v is not None}))
    for source, layer in layers:
        for key, raw in layer.items():
            try:
                values[key] = _parse(key, raw)
            except ValueError as e:
                raise SettingsError(f"{source}: {e}") from None
    return Settings(**{FIELDS[k]: v for k, v in values.items()},
                    model_layers=model_layers, route_layers=route_layers)
