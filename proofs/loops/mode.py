"""The auto/human mode and the hotkey that toggles it mid-run."""

from __future__ import annotations

# "auto" runs unattended: a lemma is picked from the shortlist at random
# (serial) or taken oldest-first from the buffer (parallel).
# "human" stops at every planning step. Set from --mode, then mutated by the
# hotkey and the menu, so it is genuinely a run-time toggle rather than a
# launch-time one.
DEFAULT_MODE = "auto"


HOTKEY_KEYS = "h"            # single keystroke; see ui.input.HotKey


def check_mode_toggle(ctx, verbose: bool = True) -> None:
    """Apply a hotkey press, if one has landed since the last check.

    Called at several points in an iteration rather than only before the menu,
    because a press lands whenever the operator changes their mind and that is
    usually somewhere in the middle of a long prover, verifier or reviser call.
    MODE is
    still only *read* at the planning step, so checking early changes nothing
    functionally — it means the log acknowledges the key at the moment you
    press it instead of twenty minutes later, which is the difference between
    a toggle that feels responsive and one you press twice because the first
    press seemed not to register.

    The press is a toggle, so it works from either mode: it is the way back to
    automation from anywhere in the loop except the menu itself, where the
    listener is paused and 'a' does the same job.
    """
    if not ctx.run.hotkey.take():
        return
    ctx.run.mode = "auto" if ctx.run.mode == "human" else "human"
    ctx.ui.log(
        "\nAutomation resumed; planning runs unattended from here."
        if ctx.run.mode == "auto"
        else "\nManual control engaged; you pick at the next planning step.",
        verbose,
    )
