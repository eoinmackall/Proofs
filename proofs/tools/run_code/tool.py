"""The run_code tool: one complete script, run in a fresh sandboxed process
(execution.py), and its status and RESULT line read back."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..base import Tool


def run_code(runner: Any, hint: str,
             on_harness_error: Callable[[str], None] = lambda _m: None,
             max_rounds: int = 8) -> Tool:
    """run_code over runner (execution.Runner): the model passes a complete
    script and its language, and reads back the run — its status, its
    RESULT, the tails of its output (execution.RunResult.to_text). hint is
    the script contract (agents/coding.md, filled in by the caller). A
    sandbox that cannot start is the harness's failure, not the model's:
    on_harness_error tells the operator, and the model is told to go on
    without code. max_rounds is the rounds a call offered it may spend
    (tools.code_rounds): a computation is written, run, read and
    corrected, so it needs more than a lookup does."""

    def handler(args: Dict[str, Any]) -> str:
        language = str(args.get("language") or "").strip()
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return 'error: pass "code", the complete script to run'
        try:
            return runner.run(language, code).to_text()
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
            "Run one complete, self-contained script in a fresh sandboxed "
            "process (no network, no files from earlier runs) and read back "
            "its status, its RESULT line and the end of its output. "
            f"Languages: {', '.join(runner.languages)}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": runner.languages},
                "code": {"type": "string"},
            },
            "required": ["language", "code"],
        },
        handler=handler,
        hint=hint,
        max_rounds=max_rounds,
    )


class RunCode:
    """The plug-in: offered, under the role's own hint, to the roles
    setup_code made a run_code for (the prover, the reviser, the verifiers)
    when the command was given --tools — and, in a parallel run, only on
    the loops that hold a code slot."""

    name = "run_code"

    def offer(self, ctx: Any, call: Any) -> Optional[Tool]:
        if not ctx.loop.code_loop:
            return None
        return ctx.code.tools.get(call.role)
