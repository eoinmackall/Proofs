"""The run's tunables, per machine: the run.*, tools.* and llm.* keys of
the workspace's .proofs-local.json.

A setting is resolved key by key, the first of these that sets it winning:

    flags                               this command only (--max-iterations,
                                        --max-proof-attempts,
                                        --max-rejected-lemmas)
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
                              must answer (lookup_lemmas)
    tools.code_rounds         the same, for a call offered run_code: a
                              computation is written, run, read and
                              corrected, so it needs more
    tools.max_lookup          lemmas one lookup_lemmas call may return
    llm.max_retries           attempts at one model call
    llm.max_compaction_passes compaction-and-resume passes one truncated call
                              may get before its partial work is handed on
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..tools.run_code import limits

LOCAL_FILENAME = limits.LOCAL_FILENAME

DEFAULTS: Dict[str, int] = {
    "run.max_iterations": -1,
    "run.max_proof_attempts": 3,
    "run.max_rejected_lemmas": 4,
    "run.max_loop_restarts": 3,
    "tools.max_rounds": 3,
    "tools.code_rounds": 8,
    "tools.max_lookup": 10,
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
    "llm.max_retries": "llm_max_retries",
    "llm.max_compaction_passes": "max_compaction_passes",
}

# The defaults of the settings that are flags only (proofs run, proofs
# repair): the lemma window's sizes (-1 is no limit), how agents call
# tools, and the context window used when nothing is probed.
DEFAULT_LEMMAS_ALL = 25
DEFAULT_MY_LEMMAS = 25
DEFAULT_LEMMAS_COMMON = 10
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
    llm_max_retries: int = DEFAULTS["llm.max_retries"]
    max_compaction_passes: int = DEFAULTS["llm.max_compaction_passes"]
    # The lemma window (see loops.prompts.lemma_window).
    lemmas_all: int = DEFAULT_LEMMAS_ALL
    my_lemmas: int = DEFAULT_MY_LEMMAS
    lemmas_common: int = DEFAULT_LEMMAS_COMMON
    # How agents call tools: "auto" or "json" (see --tool-mode).
    tool_mode: str = DEFAULT_TOOL_MODE
    # The context window in tokens: the probed server's, or --num-ctx.
    num_ctx: int = DEFAULT_NUM_CTX

    def unlimited_iterations(self) -> bool:
        return self.max_iterations < 0


# The smallest value each key takes; run.max_iterations also takes -1.
_MINIMUM: Dict[str, int] = {"run.max_rejected_lemmas": 0}


class SettingsError(Exception):
    """The file is unreadable, or a value is not usable."""


def _parse(key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{key} must be an integer, not {value!r}")
    try:
        n = int(value)
    except ValueError:
        raise ValueError(f"{key} must be an integer, not {value!r}") from None
    if key == "run.max_iterations" and n == -1:
        return n
    low = _MINIMUM.get(key, 1)
    if n < low:
        extra = " (or -1, for no limit)" if key == "run.max_iterations" else ""
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
