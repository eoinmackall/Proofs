"""Resource limits for code the agents run (execution.py), per machine.

A limit is resolved key by key, the first of these that sets it winning:

    --code-* flags                       this command only
    conjectures/NAME/limits.json         this conjecture on this machine
    <workspace>/.proofs-local.json       this machine
    DEFAULTS                             built in

Both files are gitignored: limits describe the machine a run is on, not
the conjecture, so they are never shared. Each is a JSON object holding
only the keys it overrides, named the way config.py names its keys:

    {"code.memory": "32G", "code.jobs": 4}

so the per-conjecture file starts empty and stays small, and a change to
the workspace file reaches every conjecture that does not override it.

The keys:

    code.timeout        seconds one execution may run (wall clock)
    code.memory         memory one execution may hold (PSS, all its processes)
    code.jobs           code job slots on this machine, shared by every proofs
                        command on it (execution.acquire_slots): one per loop
                        that may run code
    code.threads        threads one execution is told to use (OMP, BLAS, Sage)
    code.max_tasks      processes and threads one execution may have at once:
                        a guard against a runaway fork, not a parallelism
                        allowance; it counts threads, so it must sit above
                        code.threads
    code.output_cap     bytes of stdout plus stderr one execution may write
    code.reserve_memory memory kept back for proofs itself, and for a local
                        llama-server if it shares the machine

Sizes take a K, M, G or T suffix (powers of 1024) or a bare byte count.

"Per execution" is per script run: one call of the run_code tool is one
fresh process tree, killed when it passes any of its limits. code.jobs is
the only machine-wide number, and code.jobs x code.memory +
code.reserve_memory must fit in physical memory — check() refuses a run
that does not, since running out of memory is what takes a machine down.
Threads beyond the CPU count only slow things down, so that is a warning.

This module is standalone: it imports nothing from this package.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

LOCAL_FILENAME = ".proofs-local.json"     # at the workspace root
CONJECTURE_FILENAME = "limits.json"       # in a conjecture directory

DEFAULTS: Dict[str, Any] = {
    "code.timeout": 600,
    "code.memory": "8G",
    "code.jobs": 1,
    "code.threads": 1,
    "code.max_tasks": 32,
    "code.output_cap": "1M",
    "code.reserve_memory": "4G",
}
SIZE_KEYS = frozenset({"code.memory", "code.output_cap", "code.reserve_memory"})
KEYS = tuple(DEFAULTS)

_SIZE_RE = re.compile(r"\s*(\d+)\s*([KMGT]?)i?B?\s*\Z", re.IGNORECASE)
_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


class LimitsError(Exception):
    """A limits file is unreadable, or a value is not usable."""


def parse_size(value: Any) -> int:
    """value as a byte count: an int, or a string like "8G"."""
    if isinstance(value, bool):
        raise ValueError(f"{value!r} is not a size")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{value!r} is not a positive size")
        return value
    m = _SIZE_RE.match(str(value))
    if not m or int(m.group(1)) <= 0:
        raise ValueError(
            f"{value!r} is not a size: use a byte count or a number with "
            f"a K, M, G or T suffix (8G)"
        )
    return int(m.group(1)) * _UNITS[m.group(2).upper()]


def format_size(n: int) -> str:
    """n bytes the way a limits file writes it: the largest exact unit."""
    for unit in ("T", "G", "M", "K"):
        if n % _UNITS[unit] == 0:
            return f"{n // _UNITS[unit]}{unit}"
    return str(n)


def human_size(n: int) -> str:
    """n bytes for a person to read: one decimal in the largest unit that
    keeps the number at least 1 (8.0G, 301.3M)."""
    for unit in ("T", "G", "M", "K"):
        if n >= _UNITS[unit]:
            return f"{n / _UNITS[unit]:.1f}{unit}"
    return f"{n}B"


def _parse(key: str, value: Any) -> int:
    if key in SIZE_KEYS:
        return parse_size(value)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{key} must be a positive integer, not {value!r}")
    try:
        n = int(value)
    except ValueError:
        raise ValueError(f"{key} must be a positive integer, not {value!r}") from None
    if n <= 0:
        raise ValueError(f"{key} must be a positive integer, not {value!r}")
    return n


@dataclass(frozen=True)
class Limits:
    timeout: int
    memory: int
    jobs: int
    threads: int
    max_tasks: int
    output_cap: int
    reserve_memory: int
    # key -> where its value came from: "default", a file path, or "flag".
    sources: Mapping[str, str]

    def value(self, key: str) -> int:
        return getattr(self, key.split(".", 1)[1])

    def describe(self) -> str:
        """One line per key: its value and where it came from."""
        lines = []
        for key in KEYS:
            v = self.value(key)
            shown = format_size(v) if key in SIZE_KEYS else str(v)
            lines.append(f"{key:<20} {shown:<8} ({self.sources[key]})")
        return "\n".join(lines)


def _read(path: Path) -> Dict[str, Any]:
    """A limits file's object, {} when it is not there."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LimitsError(f"{path} is not a readable limits file: {e}") from e
    if not isinstance(data, dict):
        raise LimitsError(f"{path} must hold a JSON object")
    unknown = sorted(set(data) - set(KEYS))
    if unknown:
        raise LimitsError(
            f"{path}: unknown key(s) {', '.join(unknown)}; the keys are "
            f"{' '.join(KEYS)}"
        )
    return data


def resolve(
    workspace_root: Optional[Path],
    conjecture_dir: Optional[Path],
    flags: Optional[Mapping[str, Any]] = None,
) -> Limits:
    """The limits in effect, key by key (see the module docstring). flags
    maps a key to the value a --code-* flag gave, None for a flag not
    passed."""
    layers: List[Tuple[str, Mapping[str, Any]]] = [("default", DEFAULTS)]
    def label(p: Path) -> str:
        if workspace_root is not None:
            try:
                return str(p.relative_to(workspace_root))
            except ValueError:
                pass
        return str(p)

    if workspace_root is not None:
        p = workspace_root / LOCAL_FILENAME
        layers.append((label(p), _read(p)))
    if conjecture_dir is not None:
        p = conjecture_dir / CONJECTURE_FILENAME
        layers.append((label(p), _read(p)))
    if flags:
        layers.append(("flag", {k: v for k, v in flags.items() if v is not None}))
    values: Dict[str, int] = {}
    sources: Dict[str, str] = {}
    for source, layer in layers:
        for key, raw in layer.items():
            try:
                values[key] = _parse(key, raw)
            except ValueError as e:
                raise LimitsError(f"{source}: {e}") from None
            sources[key] = source
    return Limits(
        **{key.split(".", 1)[1]: values[key] for key in KEYS},
        sources=sources,
    )


def physical_memory() -> Optional[int]:
    """This machine's physical memory in bytes, None when it cannot be read."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def total_memory(limits: Limits) -> int:
    """What a run may hold at most: every job at its cap, plus the reserve."""
    return limits.jobs * limits.memory + limits.reserve_memory


def total_tasks(limits: Limits) -> int:
    """The tasks a run may have at most: every job at its cap, plus room
    for proofs' own threads."""
    return limits.jobs * limits.max_tasks + 256


def check(limits: Limits, overcommit: bool = False) -> Tuple[List[str], List[str]]:
    """(errors, warnings) for running under limits on this machine. An
    error stops the run; overcommit turns the memory error into a warning."""
    errors: List[str] = []
    warnings: List[str] = []
    ram = physical_memory()
    need = total_memory(limits)
    if ram is not None and need > ram:
        msg = (
            f"code.jobs x code.memory + code.reserve_memory = "
            f"{limits.jobs} x {format_size(limits.memory)} + "
            f"{format_size(limits.reserve_memory)} = {format_size(need)}, "
            f"more than this machine's {human_size(ram)} of memory"
        )
        if overcommit:
            warnings.append(msg + " (--code-overcommit)")
        else:
            errors.append(
                msg + ". Lower code.jobs or code.memory, or pass "
                "--code-overcommit to run anyway."
            )
    cpus = os.cpu_count()
    if cpus is not None and limits.jobs * limits.threads > cpus:
        warnings.append(
            f"code.jobs x code.threads = {limits.jobs * limits.threads} is "
            f"more than this machine's {cpus} hardware threads: executions "
            f"running together will slow each other down"
        )
    if limits.max_tasks < limits.threads + 8:
        warnings.append(
            f"code.max_tasks ({limits.max_tasks}) leaves little room above "
            f"code.threads ({limits.threads}): Sage's helper processes count "
            f"against it too"
        )
    return errors, warnings
