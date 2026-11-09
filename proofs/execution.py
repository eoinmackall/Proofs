"""Running an agent's script: one fresh, limited process tree per call.

The run_code tool (tools.py) hands a complete script here — Pi-style: the
model writes the whole program, never a REPL session, so no state carries
from one call to the next and the script that ran is exactly the script on
disk. Runner.run() writes it under the user's scratch directory, runs it
inside the inner sandbox (sandbox.py) with an environment built from
nothing, polices it with a watchdog, and returns what happened.

Each call gets a run directory, gitignored:

    conjectures/NAME/scratch/<user_id>/<run_id>/
        script.sage | script.py   the script, exactly as it ran
        stdout.txt, stderr.txt    its output, cut at code.output_cap
        run.json                  the run log (below)

and a work/ directory beside them — the one place the script may write,
mounted as /work — which is deleted when the run ends.

The script contract (agents/coding.md tells the agents): the script's
asserts encode what it checks, so exit status 0 means the claim held, and
it ends by printing exactly one line

    RESULT: <json>

which the harness parses into the run's result, capped at tools.lock's
policy.result_bytes. Anything else on stdout is log.

The watchdog polls the run's process tree every POLL_SECONDS: wall time
against code.timeout, the tree's memory (PSS, so shared libraries are
counted once) against code.memory, its processes and threads against
code.max_tasks, and the output files against code.output_cap. Past any of
them the whole tree is killed — with the inner sandbox, killing bwrap
kills the PID namespace and everything in it. code.jobs bounds how many
runs execute at once in this process.

Every failure comes back as a message the model can act on (a timeout, an
assert's traceback, a missing RESULT line); only a sandbox that cannot
start is the harness's error, raised as HarnessError for the operator.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

try:  # inside the proofs package
    from . import codetools, limits as limits_mod, sandbox
except ImportError:  # top-level modules
    import codetools
    import limits as limits_mod
    import sandbox

POLL_SECONDS = 0.25
MAX_SCRIPT_BYTES = 256 * 1024
# How much of stdout and stderr the model is shown, at most, in lines.
TAIL_LINES = 40
RESULT_PREFIX = "RESULT:"

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


class HarnessError(Exception):
    """The run could not be started: the harness's failure, not the script's."""


# Every execution in flight in this process, so an interrupt can kill them
# all (kill_all) before proofs exits: executions run in their own session,
# where the terminal's Ctrl-C never reaches them.
_LIVE: Set[subprocess.Popen] = set()
_LIVE_LOCK = threading.Lock()


def kill_all() -> None:
    """Kill every execution in flight. Safe to call from a signal handler
    and more than once."""
    with _LIVE_LOCK:
        procs = list(_LIVE)
    for p in procs:
        _kill(p)


# Unsandboxed executions do not die with proofs on their own.
atexit.register(kill_all)


def _kill(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            p.kill()
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Machine-wide job slots
# ----------------------------------------------------------------------------
# code.jobs is a limit on the machine, not on one proofs process: two runs in
# two terminals must not each start code.jobs executions. A slot is an
# exclusive flock on slot-<i>.lock in slot_dir(), i < code.jobs, held for the
# life of the process (the kernel lets go when it exits, however it exits).
# A process takes one slot per loop that may run code.
_SLOTS: List[int] = []


def slot_dir() -> Path:
    """Where the slot locks live: per user, per machine, never in a
    workspace. $XDG_RUNTIME_DIR when there is one (cleared at logout),
    else a directory under the system temp dir."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    if base and os.path.isdir(base):
        return Path(base) / "proofs" / "slots"
    import tempfile
    return Path(tempfile.gettempdir()) / f"proofs-{os.getuid()}" / "slots"


def acquire_slots(jobs: int, wanted: int) -> int:
    """Take up to wanted of the machine's jobs slots, without waiting.
    Returns how many this process now holds (0 when every slot is taken)."""
    try:
        import fcntl
    except ImportError:  # no flock (Windows): this process's own count only
        return min(jobs, wanted)
    d = slot_dir()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    for i in range(jobs):
        if len(_SLOTS) >= wanted:
            break
        fd = os.open(d / f"slot-{i}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        _SLOTS.append(fd)
    return len(_SLOTS)


# ----------------------------------------------------------------------------
# The process tree, from /proc
# ----------------------------------------------------------------------------

def _proc_table() -> Dict[int, tuple]:
    """pid -> (ppid, threads) for every process /proc shows."""
    table: Dict[int, tuple] = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return table
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                stat = f.read().decode("ascii", "replace")
        except OSError:
            continue
        rest = stat[stat.rfind(")") + 2:].split()
        # rest[0] is the state; ppid is rest[1], num_threads rest[17].
        try:
            table[int(name)] = (int(rest[1]), int(rest[17]))
        except (IndexError, ValueError):
            continue
    return table


def _tree(root: int, table: Mapping[int, tuple]) -> List[int]:
    children: Dict[int, List[int]] = {}
    for pid, (ppid, *_rest) in table.items():
        children.setdefault(ppid, []).append(pid)
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        if pid in table:
            out.append(pid)
        todo.extend(children.get(pid, ()))
    return out


def _memory(pid: int) -> int:
    """pid's proportional set size in bytes (its resident size where
    smaps_rollup cannot be read)."""
    try:
        with open(f"/proc/{pid}/smaps_rollup", "rb") as f:
            for line in f:
                if line.startswith(b"Pss:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open(f"/proc/{pid}/statm", "rb") as f:
            return int(f.read().split()[1]) * _PAGE
    except (OSError, ValueError, IndexError):
        return 0


def _ps_table() -> Dict[int, tuple]:
    """pid -> (ppid, 1, rss bytes) from ps, where there is no /proc
    (macOS): processes only, each counted as one task, resident memory."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    table: Dict[int, tuple] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            table[int(parts[0])] = (int(parts[1]), 1, int(parts[2]) * 1024)
    return table


def _usage(root: int) -> tuple:
    """(memory, tasks) of root's process tree: from /proc (PSS, threads
    counted), else from ps (resident size, processes only)."""
    if os.path.isdir("/proc/self"):
        table = _proc_table()
        pids = _tree(root, table)
        return (sum(_memory(p) for p in pids), sum(table[p][1] for p in pids))
    table = _ps_table()
    pids = _tree(root, table)
    return (sum(table[p][2] for p in pids), len(pids))


# ----------------------------------------------------------------------------
# A run's outcome
# ----------------------------------------------------------------------------

@dataclass
class RunResult:
    run_id: str
    language: str
    script_sha256: str
    status: str                       # "ok" or "failed"
    failure: str = ""                 # what failed, as the model is told
    # The watchdog killed it at a limit (timeout, memory, tasks, output):
    # a fact about this machine's limits, not about the script's claim.
    limit_killed: bool = False
    exit_code: Optional[int] = None
    result: Any = None                # the parsed RESULT, when status is ok
    wall_seconds: float = 0.0
    peak_memory: int = 0
    peak_tasks: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    log: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """The run as the model reads it."""
        lines = [
            f"run_id: {self.run_id}",
            f"script id: {self.script_sha256[:8]}  (refer to it as "
            f"[code:{self.script_sha256[:8]}])",
            f"status: {self.status}" + (f" — {self.failure}" if self.failure else ""),
            f"exit code: {self.exit_code}  (wall {self.wall_seconds:.1f} s, "
            f"peak memory {limits_mod.human_size(self.peak_memory)})",
        ]
        if self.status == "ok":
            lines.append(f"RESULT: {json.dumps(self.result, sort_keys=True)}")
        if self.stdout_tail:
            lines += ["--- stdout (tail) ---", self.stdout_tail]
        if self.stderr_tail:
            lines += ["--- stderr (tail) ---", self.stderr_tail]
        return "\n".join(lines)


def _tail(path: Path, n: int = TAIL_LINES) -> str:
    """The last n lines of path, less the RESULT line (shown on its own)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [l for l in text.rstrip("\n").splitlines()
             if not l.startswith(RESULT_PREFIX)]
    if len(lines) <= n:
        return "\n".join(lines)
    return f"[... {len(lines) - n} earlier lines]\n" + "\n".join(lines[-n:])


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def normalize_script(code: str) -> str:
    """A script as it is stored and hashed: line endings to \\n, nothing
    else touched (code is whitespace-sensitive)."""
    return code.replace("\r\n", "\n").replace("\r", "\n")


def parse_result(stdout: str, cap: int) -> tuple:
    """(result, failure) from a script's stdout: the one RESULT line's
    JSON and "", or None and what is wrong with it."""
    found = [l for l in stdout.splitlines() if l.startswith(RESULT_PREFIX)]
    if not found:
        return None, (
            "the script printed no RESULT line. End it with exactly one "
            "line: print(\"RESULT: \" + json.dumps(...)), converting Sage "
            "values with int(), str() or list() first."
        )
    if len(found) > 1:
        return None, f"the script printed {len(found)} RESULT lines; print exactly one."
    text = found[0][len(RESULT_PREFIX):].strip()
    size = len(text.encode("utf-8"))
    if size > cap:
        return None, (
            f"the RESULT line is {size:,} bytes; tools.lock allows {cap:,}. "
            f"Print a summary instead of the object: counts, a few "
            f"examples, or a SHA-256 of a canonical (sorted) serialization."
        )
    try:
        return json.loads(text), ""
    except ValueError as e:
        return None, f"the RESULT line is not valid JSON ({e})."


# ----------------------------------------------------------------------------
# The runner
# ----------------------------------------------------------------------------

class Runner:
    """Runs scripts for one proofs process: the tools it may use, under
    the limits it was given, writing under one user's scratch directory."""

    def __init__(
        self,
        tools: Mapping[str, "codetools.Found"],
        limits: "limits_mod.Limits",
        policy: Mapping[str, int],
        scratch_dir: Path,
        user_id: str,
        backend: Optional["sandbox.Backend"],
        log: Callable[[str], None] = lambda _m: None,
    ):
        # Keyed by language, the name an agent passes to run_code.
        self.tools = {f.spec.language: f for f in tools.values()}
        self.limits = limits
        self.policy = dict(policy)
        self.scratch_dir = scratch_dir
        self.user_id = user_id
        self.backend = backend        # None: unsandboxed (--unsandboxed-code)
        self.log = log
        self._slots = threading.BoundedSemaphore(limits.jobs)

    @property
    def languages(self) -> List[str]:
        return sorted(self.tools)

    def _env(self, work_dir: Path) -> Dict[str, str]:
        """The execution's whole environment: nothing is inherited, so no
        API key or token in proofs' environment reaches a script."""
        home = (self.backend.home(work_dir) if self.backend
                else str(work_dir / ".home"))
        tool_dirs = [str(Path(f.path).parent) for f in self.tools.values()]
        threads = str(self.limits.threads)
        env = {
            "PATH": sandbox.inner_path(tool_dirs) if self.backend
            else os.pathsep.join([*tool_dirs, os.environ.get("PATH", "")]),
            "HOME": home,
            "DOT_SAGE": f"{home}/.sage",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            # Fixed string hashing, so set and dict order repeat run to run.
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "SAGE_NUM_THREADS": threads,
        }
        if self.backend is None or self.backend.name != "bwrap":
            # No private /tmp without bwrap: temp files go in the work dir.
            env["TMPDIR"] = str(work_dir / ".home")
        return env

    def run(self, language: str, code: str) -> RunResult:
        found = self.tools.get(language)
        if found is None:
            raise ValueError(
                f"no tool for language {language!r}; this run has "
                f"{', '.join(self.languages)}"
            )
        script = normalize_script(code)
        data = script.encode("utf-8")
        if not data.strip():
            raise ValueError("the script is empty")
        if len(data) > MAX_SCRIPT_BYTES:
            raise ValueError(
                f"the script is {len(data):,} bytes; at most "
                f"{MAX_SCRIPT_BYTES:,} are run"
            )
        sha = hashlib.sha256(data).hexdigest()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{sha[:8]}-{uuid.uuid4().hex[:4]}"
        run_dir = self.scratch_dir / run_id
        work_dir = run_dir / "work"
        work_dir.mkdir(parents=True)
        name = "script" + found.spec.extension
        (run_dir / name).write_bytes(data)
        (work_dir / name).write_bytes(data)
        if self.backend is None or self.backend.name != "bwrap":
            (work_dir / ".home").mkdir()

        script_path = (self.backend.script_path(work_dir, name) if self.backend
                       else str(work_dir / name))
        command = found.spec.command(found.path, script_path)
        if self.backend:
            command = self.backend.wrap(work_dir, [found.prefix], command)
        with self._slots:
            try:
                outcome = self._execute(command, work_dir, run_dir)
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
        return self._finish(run_id, language, sha, found, run_dir, outcome)

    def _execute(self, command: Sequence[str], work_dir: Path,
                 run_dir: Path) -> Dict[str, Any]:
        lim = self.limits
        out_path, err_path = run_dir / "stdout.txt", run_dir / "stderr.txt"
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        kill_reason = ""
        peak_mem = peak_tasks = 0
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            try:
                p = subprocess.Popen(
                    list(command), stdin=subprocess.DEVNULL, stdout=out,
                    stderr=err, env=self._env(work_dir),
                    cwd=(self.backend.cwd(work_dir) if self.backend
                         else str(work_dir)),
                    start_new_session=True,
                )
            except OSError as e:
                raise HarnessError(f"could not start {command[0]}: {e}") from e
            with _LIVE_LOCK:
                _LIVE.add(p)
            try:
                while p.poll() is None:
                    elapsed = time.monotonic() - started
                    mem, tasks = _usage(p.pid)
                    peak_mem, peak_tasks = max(peak_mem, mem), max(peak_tasks, tasks)
                    try:
                        written = out_path.stat().st_size + err_path.stat().st_size
                    except OSError:
                        written = 0
                    if elapsed > lim.timeout:
                        kill_reason = (
                            f"timeout: killed after {lim.timeout} s "
                            f"(code.timeout). Make the computation smaller, "
                            f"or split it into several runs."
                        )
                    elif mem > lim.memory:
                        kill_reason = (
                            f"out of memory: killed at "
                            f"{limits_mod.human_size(mem)} (code.memory is "
                            f"{limits_mod.format_size(lim.memory)}). Store "
                            f"less: stream through candidates instead of "
                            f"building them all, or shrink the search."
                        )
                    elif tasks > lim.max_tasks:
                        kill_reason = (
                            f"too many processes: killed at {tasks} processes "
                            f"and threads (code.max_tasks is {lim.max_tasks})."
                        )
                    elif written > lim.output_cap:
                        kill_reason = (
                            f"too much output: killed after "
                            f"{limits_mod.human_size(written)} "
                            f"(code.output_cap is "
                            f"{limits_mod.format_size(lim.output_cap)}). "
                            f"Print less: the RESULT line and a short summary."
                        )
                    if kill_reason:
                        _kill(p)
                        p.wait()
                        break
                    time.sleep(POLL_SECONDS)
            finally:
                if p.poll() is None:
                    _kill(p)
                    p.wait()
                with _LIVE_LOCK:
                    _LIVE.discard(p)
        return {
            "returncode": p.returncode,
            "kill_reason": kill_reason,
            "wall_seconds": time.monotonic() - started,
            "started_at": started_at,
            "peak_memory": peak_mem,
            "peak_tasks": peak_tasks,
        }

    def _finish(self, run_id: str, language: str, sha: str,
                found: "codetools.Found", run_dir: Path,
                outcome: Mapping[str, Any]) -> RunResult:
        out_path, err_path = run_dir / "stdout.txt", run_dir / "stderr.txt"
        rc = outcome["returncode"]
        stderr_tail = _tail(err_path)
        if self.backend and rc != 0 and not outcome["kill_reason"] and \
                stderr_tail.startswith(("bwrap:", "sandbox-exec:")):
            raise HarnessError(f"the sandbox failed to start: {stderr_tail}")
        result, failure = None, outcome["kill_reason"]
        if not failure:
            if rc != 0:
                failure = (
                    f"killed by signal {-rc}" if rc < 0
                    else f"exited with code {rc}"
                ) + " — see stderr below (a failed assert means the claim did not hold)."
            else:
                try:
                    stdout = out_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    stdout = ""
                result, failure = parse_result(
                    stdout, self.policy["result_bytes"]
                )
        status = "failed" if failure else "ok"
        log = {
            "run_id": run_id,
            "user_id": self.user_id,
            "language": language,
            "script_sha256": sha,
            "tool": found.fingerprint(),
            "platform": {"system": platform.system(),
                         "release": platform.release(),
                         "machine": platform.machine()},
            "sandbox": self.backend.name if self.backend else "none",
            "limits": {k: self.limits.value(k) for k in limits_mod.KEYS},
            "started_at": outcome["started_at"],
            "wall_seconds": round(outcome["wall_seconds"], 3),
            "peak_memory": outcome["peak_memory"],
            "peak_tasks": outcome["peak_tasks"],
            "exit_code": rc,
            "status": status,
            "failure": failure,
            "result": result,
            "stdout_sha256": _sha256_file(out_path),
            "stderr_sha256": _sha256_file(err_path),
        }
        (run_dir / "run.json").write_text(
            json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return RunResult(
            run_id=run_id, language=language, script_sha256=sha,
            status=status, failure=failure, exit_code=rc, result=result,
            limit_killed=bool(outcome["kill_reason"]),
            wall_seconds=outcome["wall_seconds"],
            peak_memory=outcome["peak_memory"],
            peak_tasks=outcome["peak_tasks"],
            stdout_tail=_tail(out_path), stderr_tail=stderr_tail, log=log,
        )
