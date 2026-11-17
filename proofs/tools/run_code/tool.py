"""The run_code tool: a script — new, or one of the agent's files edited —
run in a fresh sandboxed process (execution.py), and its status and RESULT
line read back. The files are the calling agent's own (files.py)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..base import Tool
from . import files


def run_code(runner: Any, hint: str, scripts: "files.Scripts",
             on_harness_error: Callable[[str], None] = lambda _m: None,
             max_rounds: int = 8, transient: bool = False) -> Tool:
    """run_code over runner (execution.Runner), writing in scripts: the
    model passes a complete script and its language (a new file, which the
    harness names), or base, one of its files, and edits to make to it (in
    place, or, with new_file, as a new file); it reads back the run — the
    file's name, its status, its RESULT, the tails of its output
    (execution.RunResult.to_text). hint is the script contract
    (agents/coding.md, filled in by the caller). A sandbox that cannot
    start is the harness's failure, not the model's: on_harness_error tells
    the operator, and the model is told to go on without code. max_rounds
    is the rounds a call offered it may spend (tools.code_rounds): a
    computation is written, run, read and corrected, so it needs more than
    a lookup does. transient: the files go when the call ends (a reviser's,
    a verifier's); the prover's stay for the proof loop to manage."""

    def handler(args: Dict[str, Any]) -> str:
        code = args.get("code")
        base = args.get("base")
        has_code = isinstance(code, str) and code.strip()
        if has_code and base:
            return ('error: pass either "code" (a new file) or "base" and '
                    '"edits" (a change to one of your files), not both')
        try:
            if has_code:
                language = str(args.get("language") or "").strip()
                name = scripts.next_name()
            elif base:
                language, code = scripts.edit(str(base), args.get("edits"))
                name = scripts.next_name() if args.get("new_file") else str(base).strip()
            else:
                return ('error: pass "code", the complete script to run, or '
                        '"base" and "edits" to change one of your files')
            return runner.run(language, code, scripts.path, name).to_text()
        except ValueError as e:
            return f"error: {e}"
        except Exception as e:
            if type(e).__name__ != "HarnessError":
                raise
            on_harness_error(str(e))
            return (
                "error: the sandbox failed to start, through no fault of "
                "the script. Do not call run_code again; answer without it."
            )

    return Tool(
        name="run_code",
        description=(
            "Run a script in a fresh sandboxed process (no network, nothing "
            "kept from earlier runs) and read back its status, its RESULT "
            "line and the end of its output. Pass language and code for a "
            "new script; the result names its file. Or pass base (a file's "
            "name) and edits ({old, new} replacements, each old appearing "
            "exactly once in the file) to change that file in place, adding "
            "new_file: true to keep it and save the change as a new file. "
            f"Languages: {', '.join(runner.languages)}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": runner.languages},
                "code": {"type": "string"},
                "base": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"old": {"type": "string"},
                                       "new": {"type": "string"}},
                        "required": ["old", "new"],
                    },
                },
                "new_file": {"type": "boolean"},
            },
        },
        handler=handler,
        hint=hint,
        max_rounds=max_rounds,
        fold_note=scripts.listing,
        on_close=scripts.remove if transient else None,
    )


class RunCode:
    """The plug-in: offered, under the role's own hint, to the roles
    setup_code made a run_code for (the prover, the reviser, the verifiers)
    when the command was given --tools — and, in a parallel run, only on
    the loops that hold a code slot. Each call gets the tool over its own
    agent's directory (files.py)."""

    name = "run_code"

    def offer(self, ctx: Any, call: Any) -> Optional[Tool]:
        if not ctx.loop.code_loop:
            return None
        make = ctx.code.tools.get(call.role)
        if make is None:
            return None
        agent = call.agent or call.role
        return make(files.Scripts(files.loop_dir(ctx) / agent),
                    transient=agent not in files.KEPT_ROLES)
