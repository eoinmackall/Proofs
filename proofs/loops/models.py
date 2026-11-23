"""The models a command talks to: the routing table for the loops' roles
(llm/routing.py), built from the settings' routing layers and the backend
flags, then connected and attached to the Context.

Two steps, so that a command can check its routing before anything slow
(proofs run checks it before the sandbox relaunch): build() is offline
and raises llm.RoutingError for an unusable config; attach() probes every
model routed to, and raises llm.ProbeError for one the probe found
unservable — unreachable, a wrong base URL, a key rejected or missing, an
unknown model — and RoutingError for a key's variable that is not set.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .. import llm
from ..workspace.settings import Settings
from .reasoning import REQUEST_TIMEOUT, ROLES, SAME_MODEL


def build(settings: Settings, flags: "llm.Flags") -> "llm.Models":
    """The routing table for every role, checked but not connected."""
    return llm.Models.build(settings.model_layers, settings.route_layers,
                            flags, ROLES, SAME_MODEL)


def attach(ctx: Any, models: "llm.Models", verbose: bool,
           roles: Optional[Sequence[str]] = None) -> None:
    """Connect the models of the roles the command calls (roles; every
    role when None), make them ctx's, and log what the probes found and,
    when more than one model serves the command, which serves which role."""
    notes = models.connect(REQUEST_TIMEOUT, roles)
    ctx.models = models
    for note in notes:
        ctx.ui.log(note, verbose)
    if models.routed():
        ctx.ui.log(models.describe(), verbose)


def connect(ctx: Any, settings: Settings, flags: "llm.Flags",
            verbose: bool, roles: Optional[Sequence[str]] = None) -> None:
    """build() then attach(): the routing for ctx, connected for roles."""
    attach(ctx, build(settings, flags), verbose, roles)
