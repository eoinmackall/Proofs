"""The run's tunables, per machine: the run.*, tools.* and llm.* keys of
the workspace's .proofs-local.json.

A setting is resolved key by key, the first of these that sets it winning:

    flags                               this command only (FLAGS names
                                        each key's; cli/flags.py adds them)
    <workspace>/.proofs-local.json      this machine
    DEFAULTS                            built in

The file is the one the code limits live in (tools/run_code/limits.py
reads its code.* keys); it is gitignored, and proofs init writes it with
every key at its default, so the machine's settings are there to edit from
the start. The keys:

    run.max_iterations        loop iterations before the run stops; -1 runs
                              until the conjecture is settled or the run is
                              interrupted. In a parallel run, per loop.
    run.max_proof_attempts    prover rounds one iteration may spend on a
                              lemma before it is given up and re-planned
    run.max_rejected_lemmas   how many of the most recently rejected lemmas
                              the planner is shown
    run.max_loop_restarts     parallel mode: restarts in a row of a crashed
                              thread before it is given up for the run
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

Every key but run.max_rejected_lemmas also takes -1, for no limit (a tool
call's rounds are then bounded by the context window alone). within,
below, counts and of_limit read a limit that may be -1.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from ..tools.run_code import limits

LOCAL_FILENAME = limits.LOCAL_FILENAME

DEFAULTS: Dict[str, int] = {
    "run.max_iterations": -1,
    "run.max_proof_attempts": 3,
    "run.max_rejected_lemmas": 4,
    "run.max_loop_restarts": 3,
    "tools.max_rounds": 8,
    "tools.code_rounds": 8,
    "tools.max_lookup": 10,
    "tools.max_search": 10,
    "llm.max_retries": 3,
    "llm.max_compaction_passes": 2,
}
KEYS = tuple(DEFAULTS)

# The Settings field each key sets.
FIELDS: Dict[str, str] = {
    "run.max_iterations": "max_iterations",
    "run.max_proof_attempts": "max_proof_attempts",
    "run.max_rejected_lemmas": "max_rejected_lemmas",
    "run.max_loop_restarts": "max_loop_restarts",
    "tools.max_rounds": "max_tool_rounds",
    "tools.code_rounds": "code_tool_rounds",
    "tools.max_lookup": "max_lookup",
    "tools.max_search": "max_search",
    "llm.max_retries": "llm_max_retries",
    "llm.max_compaction_passes": "max_compaction_passes",
}

# The defaults of the settings that are flags only (proofs run, proofs
# repair): how the lemma window is chosen (loops/strategies.py) and its
# size (-1 is no limit), how agents call tools, and the context window
# used when nothing is probed.
DEFAULT_WINDOW_SIZE = 30
# The window's strategies; the two that take a dial (random's beta,
# depth's jitter), a number from 0 to 1, after a comma: random,0.5.
STRATEGIES = ("recent", "random", "depth", "breadth", "relevance", "popular")
DIALS = ("random", "depth")
DEFAULT_STRATEGY: Tuple[str, float] = ("recent", 0.0)
DEFAULT_TOOL_MODE = "auto"
DEFAULT_NUM_CTX = 65536


@dataclass
class Settings:
    """The settings a command runs with: the keys above as fields (FIELDS
    names them), and the flag-only ones."""

    max_iterations: int = DEFAULTS["run.max_iterations"]
    max_proof_attempts: int = DEFAULTS["run.max_proof_attempts"]
    max_rejected_lemmas: int = DEFAULTS["run.max_rejected_lemmas"]
    max_loop_restarts: int = DEFAULTS["run.max_loop_restarts"]
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
    # The context window in tokens: the probed server's, or --num-ctx.
    num_ctx: int = DEFAULT_NUM_CTX
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
        raise ValueError(f"{key} must be at least {low}{extra}, not {n}")
    return n


def read(workspace_root: Optional[Path]) -> Dict[str, Any]:
    """The settings keys .proofs-local.json holds, {} when there is no
    file. Its code.* keys are limits.py's; any other key is an error."""
    if workspace_root is None:
        return {}
    path = workspace_root / LOCAL_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SettingsError(f"{path} is not readable: {e}") from e
    if not isinstance(data, dict):
        raise SettingsError(f"{path} must hold a JSON object")
    unknown = sorted(k for k in data
                     if k not in DEFAULTS and not k.startswith("code."))
    if unknown:
        raise SettingsError(
            f"{path}: unknown key(s) {', '.join(unknown)}; the keys are "
            f"{' '.join(KEYS)} and the code.* limits"
        )
    return {k: v for k, v in data.items() if k in DEFAULTS}


def resolve(
    workspace_root: Optional[Path],
    flags: Optional[Mapping[str, Any]] = None,
) -> Settings:
    """The settings in effect (see the module docstring). flags maps a key
    to the value a flag gave, None for a flag not passed."""
    values: Dict[str, int] = {}
    layers = [("default", DEFAULTS), (LOCAL_FILENAME, read(workspace_root))]
    if flags:
        layers.append(("flag", {k: v for k, v in flags.items() if v is not None}))
    for source, layer in layers:
        for key, raw in layer.items():
            try:
                values[key] = _parse(key, raw)
            except ValueError as e:
                raise SettingsError(f"{source}: {e}") from None
    return Settings(**{FIELDS[k]: v for k, v in values.items()})
