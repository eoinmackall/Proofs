"""--tools: checking the code tools and limits, the sandbox relaunch, and
the run_code tool each role is offered.
"""

from __future__ import annotations

import argparse
import functools
import re
import signal
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.context import Code
from ...core.dag import load_file
from ...ui.plain import PlainUI
from ...workspace import config
from ...workspace import layout as workspace
from ...workspace import settings as settings_mod
from . import codetools, execution
from . import limits as limits_mod
from . import sandbox
from .tool import run_code


def add_code_flags(parser: argparse.ArgumentParser) -> None:
    """--tools and the --code-* limits (limits.py), for the commands whose
    agents may run code."""
    parser.add_argument(
        "--tools", default=None, metavar="LIST",
        help=(
            "Let the prover and the reviser run code: a comma-separated "
            "list of tools from the conjecture's tools.lock (sage, python), "
            "or all. Each must be on PATH at the declared version. The "
            "whole run is then sandboxed (Linux, bubblewrap)."
        ),
    )
    for key in limits_mod.KEYS:
        name = key.split(".", 1)[1].replace("_", "-")
        parser.add_argument(
            f"--code-{name}", dest=f"code_{key.split('.', 1)[1]}",
            default=None, metavar="N",
            help=f"Override {key} for this run (default "
                 f"{limits_mod.DEFAULTS[key]}; see `proofs limits`).",
        )
    parser.add_argument(
        "--code-overcommit", action="store_true",
        help="Run even when code.jobs x code.memory + code.reserve_memory "
             "exceeds this machine's memory.",
    )
    parser.add_argument(
        "--unsandboxed-code", action="store_true",
        help="Run code without the sandbox: only where bubblewrap is "
             "unavailable, and only for code you would run yourself.",
    )


def _prompt_sections(text: str) -> Tuple[str, Dict[str, str]]:
    """(the text before the first "## name" heading, {name: section}) —
    how agents/coding.md holds one shared contract and the parts for each
    role."""
    shared: List[str] = []
    sections: Dict[str, List[str]] = {}
    current: Optional[List[str]] = None
    for line in text.splitlines():
        m = re.match(r"##\s+(\w+)\s*$", line)
        if m:
            current = sections.setdefault(m.group(1).lower(), [])
        elif current is None:
            shared.append(line)
        else:
            current.append(line)
    return ("\n".join(shared).strip(),
            {k: "\n".join(v).strip() for k, v in sections.items()})


def code_limit_flags(args: argparse.Namespace) -> Dict[str, Any]:
    """The --code-* limit flags as limits.resolve() takes them."""
    return {key: getattr(args, f"code_{key.split('.', 1)[1]}", None)
            for key in limits_mod.KEYS}


def setup_code(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    root: Optional[Path],
    conjecture: Optional[str] = None,
    ui: Any = None,
) -> Code:
    """For a run given --tools: check the tools and the limits, relaunch
    the whole command inside the sandbox (sandbox.relaunch, which does not
    return) unless this is already the relaunched process, and set
    return the Code (runner, tools, answer_note). Every problem is a usage
    error, reported before anything runs. proofs run calls it, and so do
    proofs verify and proofs repair (cli/main.py), with conjecture naming the
    directory their DAG or refutation file belongs to."""
    if sandbox.inside():
        # The launcher ignored SIGINT so bwrap would survive a Ctrl-C (see
        # sandbox.py); an ignored signal stays ignored across exec, so put
        # back the default for the stretch before main() installs its own.
        signal.signal(signal.SIGINT, signal.default_int_handler)
    ui = ui or PlainUI()
    code = Code()
    if args.tools is None:
        return code
    parallel = getattr(args, "parallel", None)
    # proofs verify / repair --parallel: a slot for each verifier that may
    # run a script at the same time as another (at most the four steps).
    verifiers = min(max(1, getattr(args, "parallel_verifiers", 1)), 4)
    try:
        user = config.ensure_user_id()
        paths = workspace.resolve(conjecture or args.conjecture, user, root)
        lock = codetools.load_lock(paths.root)
        names = codetools.requested(args.tools, lock, paths.root)
        found = {n: codetools.find(n, lock, paths.root) for n in names}
        workspace_root = workspace.find_root(paths.root) or paths.root
        lim = limits_mod.resolve(workspace_root, paths.root,
                                 code_limit_flags(args))
        code_rounds = settings_mod.resolve(workspace_root).code_tool_rounds
    except (config.UserConfigError, workspace.ConjectureNotFound,
            codetools.ToolError, limits_mod.LimitsError,
            settings_mod.SettingsError) as e:
        parser.error(str(e))
        return
    errors, warnings = limits_mod.check(lim, overcommit=args.code_overcommit)
    if errors:
        parser.error("\n".join(errors))
    backend: Optional["sandbox.Backend"] = None
    if not args.unsandboxed_code:
        try:
            outer = sandbox.check().outer
        except sandbox.SandboxError as e:
            parser.error(str(e))
        if outer and not sandbox.inside():
            for w in warnings:
                ui.log(f"warning: {w}", err=True)
            try:
                sandbox.relaunch(
                    workspace_root,
                    [config.config_path().parent, *(f.prefix for f in found.values())],
                    limits_mod.total_memory(lim),
                    limits_mod.total_tasks(lim),
                    writable=[execution.slot_dir()],
                )
            except sandbox.SandboxError as e:
                parser.error(str(e))
        try:
            backend = sandbox.check()
        except sandbox.SandboxError as e:
            parser.error(f"inside the sandbox: {e}")
        if not outer:
            for w in warnings:
                ui.log(f"warning: {w}", err=True)
            ui.log(f"note: {backend.name} confines each script; proofs itself "
                   f"runs unconfined here (a sandbox cannot be nested)",
                   err=True)
    else:
        for w in warnings:
            ui.log(f"warning: {w}", err=True)
        ui.log("warning: --unsandboxed-code: scripts the agents write run "
               "with your permissions, network and files", err=True)
    # The machine-wide job slots: one per loop that may run code, at most
    # code.jobs across every proofs process on this machine. A serial run
    # (or verify, or repair) needs its one; a parallel run takes what it
    # can, and its other loops run without code. Parallel verifiers take
    # what they can too, at least one: with fewer slots than verifiers,
    # their scripts wait their turn (Runner's semaphore).
    wanted = max(1, parallel) if parallel is not None else verifiers
    try:
        code.loops = execution.acquire_slots(lim.jobs, wanted)
    except OSError as e:
        parser.error(f"could not take a code job slot in {execution.slot_dir()}: {e}")
    if code.loops == 0:
        busy = (f"all code.jobs = {lim.jobs} code slot(s) on this machine are "
                f"held by other proofs commands")
        if parallel is None:
            parser.error(busy + "; wait for one to finish, or raise code.jobs")
        ui.log(f"warning: {busy}: every loop runs without code", err=True)
    runner = execution.Runner(
        found, lim, lock.policy,
        paths.root / workspace.SCRATCH_DIRNAME / user, user, backend,
        slots=code.loops,
    )
    text = string.Template(load_file(str(paths.prompts["coding.md"]))).safe_substitute(
        languages=", ".join(runner.languages),
        timeout=lim.timeout,
        memory=limits_mod.format_size(lim.memory),
        threads=lim.threads,
        result_bytes=lock.policy["result_bytes"],
    )
    shared, sections = _prompt_sections(text)
    code.runner = runner
    for role in ("prover", "reviser", "verifier"):
        hint = shared + ("\n\n" + sections[role] if sections.get(role) else "")
        code.tools[role] = functools.partial(
            run_code, runner, hint,
            on_harness_error=lambda m: ui.log(f"run_code: {m}"),
            max_rounds=code_rounds,
        )
    code.answer_note = sections.get("answer", "")
    tools_desc = ", ".join(f"{f.spec.name} {f.version}" for f in found.values())
    if parallel is not None:
        tools_desc += (f"; {code.loops} of {max(1, parallel)} loop(s) may "
                       f"run code")
    elif verifiers > 1:
        tools_desc += (f"; {code.loops} script(s) at once across "
                       f"{verifiers} parallel verifiers"
                       + (f" (the slots free of code.jobs = {lim.jobs} on "
                          f"this machine)" if code.loops < verifiers else ""))
    ui.log(
        f"code tools: {tools_desc} "
        f"({'sandboxed: ' + backend.name if backend else 'UNSANDBOXED'}; {lim.timeout} s, "
        f"{limits_mod.format_size(lim.memory)}, {lim.threads} thread(s) "
        f"per run; scripts under {runner.scratch_dir})",
        err=True,
    )
    return code
