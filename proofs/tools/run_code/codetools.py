"""The code tools an agent may run (Sage, Python), and tools.lock.

A conjecture's maintainer declares which tools its runs may use, and at
which versions, in a committed file beside conjecture.md:

    conjectures/NAME/tools.lock

    {
      "tools":  {"sage": {"version": "10.4"},
                 "python": {"version": "3.13.5", "packages": {"sympy": "1.13.3"}}},
      "policy": {"result_bytes": 65536}
    }

The maintainer does not write it by hand: the conjecture's flake.nix, which
the maintainer also owns, exports the same object as its proofsTools
output, and `proofs tools lock NAME` writes it out with `nix eval` (which
evaluates the flake and runs none of it — no shellHook, no build outside
Nix's own sandbox). Everyone else reads only tools.lock, so a user with a
system-wide Sage of the right version and no Nix at all is served the
same way as one inside `nix develop`.

A run asks for tools with --tools (a comma-separated list, or all), and
find() checks each one: declared in tools.lock, an executable on PATH, and
reporting the declared version. A declared version matches a reported one
that equals it or extends it at a dot ("10.4" matches 10.4 and 10.4.1).
Python packages are recorded in the lock but not yet checked.

The policy is shared, unlike the per-machine limits (limits.py), because
it shapes what lands in the DAG: result_bytes caps the RESULT line every
script ends with.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

LOCK_FILENAME = "tools.lock"
FLAKE_OUTPUT = "proofsTools"
DEFAULT_POLICY: Dict[str, int] = {"result_bytes": 65536}
VERSION_TIMEOUT = 120


@dataclass(frozen=True)
class ToolSpec:
    name: str                 # what --tools and tools.lock call it
    executable: str           # looked up on PATH
    language: str             # what an agent passes to run_code
    extension: str            # the script's file extension
    version_args: List[str]
    version_re: str           # group 1 is the version
    # argv to run a script, given the executable and the script's path
    command: Callable[[str, str], List[str]]


TOOLS: Dict[str, ToolSpec] = {
    "sage": ToolSpec(
        name="sage", executable="sage", language="sage", extension=".sage",
        version_args=["--version"],
        version_re=r"SageMath version ([0-9][^\s,]*)",
        # sage preparses script.sage into script.sage.py beside it, so the
        # script's directory must be writable.
        command=lambda exe, script: [exe, script],
    ),
    "python": ToolSpec(
        name="python", executable="python3", language="python",
        extension=".py",
        version_args=["--version"],
        version_re=r"Python ([0-9][^\s]*)",
        # -I: no user site-packages, no PYTHONPATH, script dir not on path
        # beyond itself; -B: no .pyc files written.
        command=lambda exe, script: [exe, "-I", "-B", script],
    ),
}


class ToolError(Exception):
    """A requested tool cannot be used: undeclared, missing, wrong version,
    or a tools.lock that cannot be read."""


@dataclass(frozen=True)
class Lock:
    tools: Mapping[str, Mapping[str, Any]]
    policy: Mapping[str, int]


@dataclass(frozen=True)
class Found:
    spec: ToolSpec
    path: str        # the executable, symlinks resolved: its fingerprint
    version: str

    @property
    def prefix(self) -> Path:
        """The directory the tool is installed under: the parent of its
        bin/ — what a sandbox must expose for the tool to run."""
        return Path(self.path).parent.parent

    def fingerprint(self) -> Dict[str, str]:
        return {"name": self.spec.name, "version": self.version, "path": self.path}


def lock_path(conjecture_dir: Path) -> Path:
    return conjecture_dir / LOCK_FILENAME


def _validate(data: Any, where: str) -> Lock:
    if not isinstance(data, dict) or not isinstance(data.get("tools"), dict):
        raise ToolError(f'{where}: expected {{"tools": {{...}}, "policy": {{...}}}}')
    tools: Dict[str, Dict[str, Any]] = {}
    for name, entry in data["tools"].items():
        if name not in TOOLS:
            raise ToolError(
                f"{where}: unknown tool {name!r}; proofs knows "
                f"{', '.join(sorted(TOOLS))}"
            )
        if not isinstance(entry, dict) or not isinstance(entry.get("version"), str) \
                or not entry["version"]:
            raise ToolError(f'{where}: tools.{name} needs a "version" string')
        tools[name] = dict(entry)
    policy = dict(DEFAULT_POLICY)
    raw_policy = data.get("policy", {})
    if not isinstance(raw_policy, dict):
        raise ToolError(f"{where}: policy must be an object")
    for key, value in raw_policy.items():
        if key not in DEFAULT_POLICY:
            raise ToolError(
                f"{where}: unknown policy key {key!r}; the keys are "
                f"{', '.join(DEFAULT_POLICY)}"
            )
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ToolError(f"{where}: policy.{key} must be a positive integer")
        policy[key] = value
    return Lock(tools=tools, policy=policy)


def load_lock(conjecture_dir: Path) -> Optional[Lock]:
    """The conjecture's tools.lock, None when it has none."""
    p = lock_path(conjecture_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ToolError(f"{p} is not a readable tools.lock: {e}") from e
    return _validate(data, str(p))


def requested(spec: str, lock: Optional[Lock], conjecture_dir: Path) -> List[str]:
    """The tool names --tools SPEC asks for, each one declared in lock."""
    if lock is None:
        raise ToolError(
            f"--tools needs {lock_path(conjecture_dir)}, which declares the "
            f"tools this conjecture's runs may use; its maintainer writes it "
            f"with `proofs tools lock {conjecture_dir.name}`"
        )
    if spec.strip() == "all":
        names = sorted(lock.tools)
    else:
        names = [n.strip() for n in spec.split(",") if n.strip()]
    if not names:
        raise ToolError("--tools: name at least one tool, or all")
    for n in names:
        if n not in lock.tools:
            raise ToolError(
                f"--tools: {n!r} is not declared in {lock_path(conjecture_dir)} "
                f"(declared: {', '.join(sorted(lock.tools)) or 'none'})"
            )
    return names


def version_matches(declared: str, reported: str) -> bool:
    return reported == declared or reported.startswith(declared + ".")


def find(name: str, lock: Lock, conjecture_dir: Path) -> Found:
    """The tool name, found on PATH at the version lock declares."""
    spec = TOOLS[name]
    declared = lock.tools[name]["version"]
    shown = os.path.relpath(conjecture_dir)
    if shown.startswith(".."):
        shown = str(conjecture_dir)
    hint = f"run `nix develop {shown}` and retry"
    exe = shutil.which(spec.executable)
    if exe is None:
        raise ToolError(
            f"{spec.executable} not found on PATH; {LOCK_FILENAME} declares "
            f"{name} {declared} — {hint}"
        )
    try:
        out = subprocess.run(
            [exe, *spec.version_args], capture_output=True, text=True,
            timeout=VERSION_TIMEOUT, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ToolError(f"{exe} {' '.join(spec.version_args)} failed: {e}") from e
    m = re.search(spec.version_re, out.stdout + out.stderr)
    if not m:
        raise ToolError(
            f"could not read a version from `{exe} "
            f"{' '.join(spec.version_args)}`"
        )
    reported = m.group(1)
    if not version_matches(declared, reported):
        raise ToolError(
            f"{name} {reported} found on PATH ({exe}), {LOCK_FILENAME} "
            f"declares {declared} — {hint}"
        )
    return Found(spec=spec, path=os.path.realpath(exe), version=reported)


def write_lock(conjecture_dir: Path) -> Path:
    """Write tools.lock from the conjecture flake's proofsTools output, by
    `nix eval`. Returns the path written."""
    if not (conjecture_dir / "flake.nix").is_file():
        raise ToolError(f"{conjecture_dir} has no flake.nix to read the tools from")
    nix = shutil.which("nix")
    if nix is None:
        raise ToolError("nix not found on PATH; tools.lock is generated from the flake by nix eval")
    ref = f"{conjecture_dir}#{FLAKE_OUTPUT}"
    try:
        out = subprocess.run(
            [nix, "--extra-experimental-features", "nix-command flakes",
             "eval", "--json", ref],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
    except OSError as e:
        raise ToolError(f"nix eval failed: {e}") from e
    if out.returncode != 0:
        raise ToolError(f"nix eval {ref} failed:\n{out.stderr.strip()}")
    try:
        data = json.loads(out.stdout)
    except ValueError as e:
        raise ToolError(f"nix eval {ref} did not return JSON: {e}") from e
    lock = _validate(data, f"{ref}")
    p = lock_path(conjecture_dir)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps({"tools": lock.tools, "policy": lock.policy},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, p)
    return p
