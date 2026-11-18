"""Tools the agents can call, and which of them an agent call is offered.

Each tool is a plug-in in a package of its own (lookup_lemmas/,
search_lemmas/, run_code/):
an object with a name and an offer(ctx, call) that returns the Tool to offer
the call described by call (Call), or None. The registry below lists them;
offered() asks each in turn, and a call offered nothing carries no tools at
all. The registry is static for now: register() is how a later entry-point
loader ("proofs.tools") will add a tool without touching this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Set

from .base import NO_ROOM, Tool, ToolSet, cut_to_budget
from .lookup_lemmas import LookupLemmas
from .run_code.tool import RunCode
from .search_lemmas import SearchLemmas

__all__ = [
    "Call", "NO_ROOM", "Tool", "ToolPlugin", "ToolSet", "cut_to_budget",
    "offered", "plugins", "register",
]

@dataclass(frozen=True)
class Call:
    """One agent call, as the tools see it when they decide whether to offer
    themselves."""

    # The agent's role: "planner", "parallel_planner",
    # "parallel_lemma_generator", "prover", "reviser", "verifier", ...
    role: str
    # The agent's own name, when its role is shared ("verifier_1", …,
    # "coding_verifier" for the role "verifier"); "" when it is the role.
    agent: str = ""
    # The complete DAG's lemmas, when the agent is shown the lemma window;
    # None when it is not (a verifier sees only what the proof cites).
    lemmas: Optional[Mapping[str, Mapping[str, Any]]] = None
    # The lemmas no agent may build on (suspension.py).
    suspended: Set[str] = field(default_factory=set)
    # The DAG snapshot (core.dag.Dag) the agent's prompt was built from,
    # when it is shown the lemma window.
    dag: Any = None
    # The lemmas the window shows in full, and how many eligible lemmas it
    # leaves out (those the tools are for).
    window: Set[str] = field(default_factory=set)
    hidden: int = 0


class ToolPlugin(Protocol):
    name: str

    def offer(self, ctx: Any, call: Call) -> Optional[Tool]:
        ...


_REGISTRY: Dict[str, ToolPlugin] = {}


def register(plugin: ToolPlugin) -> None:
    """Add a tool to the registry (a later one of the same name replaces
    an earlier one)."""
    _REGISTRY[plugin.name] = plugin


def plugins() -> List[ToolPlugin]:
    return list(_REGISTRY.values())


def offered(ctx: Any, call: Call) -> Optional[ToolSet]:
    """The tools call is offered, in registry order, or None when no tool
    offers itself — so a call with nothing to look up and no code to run
    carries no tools."""
    tools = [t for t in (p.offer(ctx, call) for p in plugins()) if t is not None]
    if not tools:
        return None
    return ToolSet(tools, default_rounds=ctx.settings.max_tool_rounds)


register(LookupLemmas())
register(SearchLemmas())
register(RunCode())
