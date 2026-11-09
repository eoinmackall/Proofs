"""Sandboxes for a run with code tools: one around proofs, one per execution.

A run given --tools has agents writing code and the harness running it, so
the whole program is confined, in two layers that need different things:

  * outer — proofs itself (relaunch()). It must reach the LLM server, so it
    keeps the network and its environment (an API key may be in it). It
    sees the filesystem read-only, $HOME hidden behind an empty tmpfs
    except what proofs needs to read (its config, its own code), and it may
    write only inside the workspace. It shares the terminal: proofs reads
    the operator's keys and handles Ctrl-C, so no --new-session here.
  * inner — one script execution (inner_argv()). No network, an environment
    built from nothing (execution.py), the system's software directories
    read-only, nothing of $HOME or the workspace, and one writable
    directory: the execution's own work directory. It runs in its own
    session (--new-session, so it cannot push keystrokes into the
    terminal) and its own PID namespace (so killing it kills everything it
    started), and dies with the harness (--die-with-parent).

Ctrl-C. A terminal sends SIGINT to its whole foreground process group,
which holds bwrap as well as proofs inside it, and bwrap dies of it at
once — taking proofs down before its handler can write the checkpoint.
So relaunch() ignores SIGINT before it execs bwrap (an ignored signal
stays ignored across exec), and proofs, inside, installs its own handlers
explicitly (main.py), which an inherited SIG_IGN does not prevent.
Executions run in their own session, out of the terminal's reach: the
harness kills them itself (execution.kill_all) on its way out.

The machine backstop. Each execution's memory and tasks are policed by the
harness's watchdog (execution.py); a runaway that outpaces its polling is
caught by one systemd scope around the whole relaunched program, capped at
code.jobs x code.memory + code.reserve_memory (limits.total_memory), where
the kernel's OOM killer picks the largest process — the runaway, not the
machine. systemd-run cannot be reached from inside the sandbox, which is
why the scope is one, outside, rather than one per execution. Without
systemd the watchdog is the only line.

On Linux both layers are bubblewrap (bwrap, from PATH or $PROOFS_BWRAP,
which the Nix package sets). On macOS the inner layer is sandbox-exec
with a generated profile (no network, writes only in the work directory,
$HOME unreadable), and there is no outer layer: a sandboxed process
cannot apply a second sandbox, so proofs itself runs unconfined there,
and the scripts — the untrusted part — are confined one by one. Without a
PID namespace, the macOS harness kills a script's process group; a script
that leaves its group with setsid() escapes the kill, as it cannot on
Linux. The macOS backend is not yet tested on a Mac. Elsewhere a run with
--tools needs --unsandboxed-code, which runs code with neither layer.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

# Set in the relaunched program: it is already inside the outer sandbox.
INSIDE_ENV = "PROOFS_SANDBOXED"
BWRAP_ENV = "PROOFS_BWRAP"

# What an execution may read: the system's software, read-only. Paths that
# do not exist on this system are skipped (--ro-bind-try).
SYSTEM_RO = (
    "/nix/store", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32",
    "/etc", "/opt", "/run/current-system",
    # macOS: the system libraries and the Homebrew prefix live here.
    "/System", "/Library",
)
# Where an execution's work directory is mounted, and its $HOME.
WORK_MOUNT = "/work"
INNER_HOME = "/tmp/home"


class SandboxError(Exception):
    """The sandbox cannot be set up on this machine."""


@dataclass(frozen=True)
class Backend:
    """How executions are confined here: "bwrap" (Linux) or
    "sandbox-exec" (macOS), and the executable that does it."""
    name: str
    exe: str

    @property
    def outer(self) -> bool:
        """Whether proofs itself runs inside a sandbox too (see the module
        docstring: macOS cannot nest one)."""
        return self.name == "bwrap"

    def script_path(self, work_dir: Path, name: str) -> str:
        """Where the execution sees its script."""
        return f"{WORK_MOUNT}/{name}" if self.name == "bwrap" else str(work_dir / name)

    def home(self, work_dir: Path) -> str:
        return INNER_HOME if self.name == "bwrap" else str(work_dir / ".home")

    def cwd(self, work_dir: Path) -> Optional[str]:
        """The cwd to start the sandbox in (bwrap sets its own)."""
        return None if self.name == "bwrap" else str(work_dir)

    def wrap(self, work_dir: Path, readable: Iterable[Path],
             command: Sequence[str]) -> List[str]:
        """argv running command, one execution, inside this sandbox."""
        if self.name == "bwrap":
            return inner_argv(self.exe, work_dir, readable, command)
        return [self.exe, "-p", macos_profile(work_dir, readable), *command]


def inside() -> bool:
    """Whether this process is the relaunched one, inside the outer sandbox."""
    return os.environ.get(INSIDE_ENV) == "1"


def bwrap() -> Optional[str]:
    return os.environ.get(BWRAP_ENV) or shutil.which("bwrap")


def _check_macos() -> Backend:
    exe = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
    if not os.path.exists(exe):
        raise SandboxError(
            "sandbox-exec not found; pass --unsandboxed-code to run code "
            "without a sandbox"
        )
    try:
        out = subprocess.run(
            [exe, "-p", "(version 1)(allow default)(deny network*)", "/usr/bin/true"],
            capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SandboxError(f"sandbox-exec could not start a sandbox: {e}") from e
    if out.returncode != 0:
        raise SandboxError(
            f"sandbox-exec could not start a sandbox here:\n{out.stderr.strip()}"
        )
    return Backend("sandbox-exec", exe)


def check() -> Backend:
    """The sandbox backend for this machine, once a trial sandbox has run;
    SandboxError with the reason when none can run here."""
    if sys.platform == "darwin":
        return _check_macos()
    if not sys.platform.startswith("linux"):
        raise SandboxError(
            f"code sandboxing needs Linux (bubblewrap) or macOS "
            f"(sandbox-exec); this is {sys.platform}. Pass "
            f"--unsandboxed-code to run code without a sandbox."
        )
    exe = bwrap()
    if exe is None:
        raise SandboxError(
            "bwrap (bubblewrap) not found on PATH; install it, or pass "
            "--unsandboxed-code to run code without a sandbox"
        )
    trial = [exe, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
             "--unshare-all", "--die-with-parent", "--",
             sys.executable, "-c", "pass"]
    try:
        out = subprocess.run(trial, capture_output=True, text=True, timeout=30,
                             stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SandboxError(f"bwrap could not start a sandbox: {e}") from e
    if out.returncode != 0:
        raise SandboxError(
            "bwrap could not start a sandbox here (unprivileged user "
            "namespaces may be disabled, or this may already be a sandbox "
            f"that forbids nesting):\n{out.stderr.strip()}"
        )
    return Backend("bwrap", exe)


def systemd_scope() -> Optional[str]:
    """systemd-run, when a user scope can be started with it."""
    exe = shutil.which("systemd-run")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--user", "--scope", "--quiet", "--", "true"],
            capture_output=True, timeout=30, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return exe if out.returncode == 0 else None


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def outer_argv(
    exe: str,
    workspace_root: Path,
    readable: Iterable[Path],
    command: Sequence[str],
    writable: Iterable[Path] = (),
) -> List[str]:
    """bwrap argv confining command (proofs) to writing in workspace_root.

    readable are the paths under $HOME proofs still needs to read — its
    config, its code, its Python's packages, the tools' install prefixes —
    which the empty tmpfs over $HOME would otherwise hide. writable are the
    few directories outside the workspace it must write: the machine-wide
    job slots (execution.slot_dir)."""
    home = Path.home().resolve()
    workspace_root = workspace_root.resolve()
    argv = [exe, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp", "--unshare-all", "--share-net"]
    if home != Path("/"):
        argv += ["--tmpfs", str(home)]
    seen = set()
    for p in sorted({Path(p).resolve() for p in readable}):
        if not p.exists() or not _under(p, home) or _under(p, workspace_root):
            continue
        if any(_under(p, s) for s in seen):
            continue
        seen.add(p)
        argv += ["--ro-bind", str(p), str(p)]
    argv += ["--bind", str(workspace_root), str(workspace_root)]
    for p in writable:
        p = Path(p).resolve()
        p.mkdir(parents=True, exist_ok=True, mode=0o700)
        argv += ["--bind", str(p), str(p)]
    cwd = Path.cwd().resolve()
    if not _under(cwd, workspace_root):
        # Relative paths on the command line are read from the cwd.
        if _under(cwd, home) and cwd not in seen:
            argv += ["--ro-bind", str(cwd), str(cwd)]
    argv += ["--chdir", str(cwd), "--", *command]
    return argv


def python_paths() -> List[Path]:
    """The directories this Python reads its own code and packages from."""
    paths = [Path(sys.prefix), Path(sys.base_prefix), Path(sys.executable).parent]
    paths += [Path(p) for p in sys.path if p]
    paths.append(Path(__file__).resolve().parent.parent)
    return paths


def relaunch(
    workspace_root: Path,
    readable: Iterable[Path],
    total_memory: int,
    total_tasks: int,
    writable: Iterable[Path] = (),
) -> None:
    """Re-exec this command inside the outer sandbox. Never returns.

    The command is rebuilt as `python -m proofs <argv>`, with PYTHONPATH
    pointing at this package so the relaunch finds the same code."""
    exe = check().exe
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env[INSIDE_ENV] = "1"
    env["PYTHONPATH"] = (
        pkg_parent + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH") else pkg_parent
    )
    # The same entry point again: proofs/main.py run as a file, or the
    # package (the proofs command, python -m proofs).
    if Path(sys.argv[0]).name == "main.py":
        command = [sys.executable, sys.argv[0], *sys.argv[1:]]
    else:
        command = [sys.executable, "-m", "proofs", *sys.argv[1:]]
    argv = outer_argv(exe, workspace_root, [*readable, *python_paths()],
                      command, writable)
    scope = systemd_scope()
    if scope is not None:
        argv = [scope, "--user", "--scope", "--quiet",
                "-p", f"MemoryMax={total_memory}", "-p", "MemorySwapMax=0",
                "-p", f"TasksMax={total_tasks}", "--", *argv]
    else:
        print("note: systemd-run is unavailable, so the only memory guard "
              "is proofs' own watchdog", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.execvpe(argv[0], argv, env)


def inner_argv(
    exe: str,
    work_dir: Path,
    readable: Iterable[Path],
    command: Sequence[str],
) -> List[str]:
    """bwrap argv for one execution of command, its script under
    WORK_MOUNT. readable are extra read-only paths (a tool installed
    outside the system directories)."""
    argv = [exe, "--unshare-all", "--new-session", "--die-with-parent"]
    for p in SYSTEM_RO:
        argv += ["--ro-bind-try", p, p]
    for p in sorted({str(Path(p).resolve()) for p in readable}):
        if not any(_under(Path(p), Path(s)) for s in SYSTEM_RO):
            argv += ["--ro-bind-try", p, p]
    argv += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
             "--dir", INNER_HOME,
             "--bind", str(work_dir), WORK_MOUNT, "--chdir", WORK_MOUNT,
             "--", *command]
    return argv


def _sbpl(path: Any) -> str:
    """path as an SBPL string literal."""
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def macos_profile(work_dir: Path, readable: Iterable[Path]) -> str:
    """The sandbox-exec profile of one execution on macOS: everything the
    default allows, except the network, any write outside the work
    directory (and the null devices), and reading $HOME — the work
    directory and the tools' own prefixes excepted. SBPL takes the last
    rule that matches, so each allow after a deny carves its exception."""
    work = Path(work_dir).resolve()
    home = Path.home().resolve()
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny file-write*)",
        f"(allow file-write* (subpath {_sbpl(work)}) (literal \"/dev/null\") "
        f"(literal \"/dev/zero\") (subpath \"/dev/fd\"))",
        f"(deny file-read* (subpath {_sbpl(home)}))",
        f"(allow file-read* (subpath {_sbpl(work)})"
        + "".join(f" (subpath {_sbpl(Path(p).resolve())})" for p in readable)
        + ")",
    ]
    return "".join(rules)


def inner_path(tool_dirs: Iterable[str]) -> str:
    """PATH for an execution: the tools' own bin directories, then the
    entries of this PATH an execution can see (under SYSTEM_RO)."""
    entries: List[str] = []
    for d in tool_dirs:
        if d not in entries:
            entries.append(d)
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d and d not in entries and any(_under(Path(d), Path(s)) for s in SYSTEM_RO):
            entries.append(d)
    return os.pathsep.join(entries)
