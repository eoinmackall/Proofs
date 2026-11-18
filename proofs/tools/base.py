"""Tools: things an agent may call mid-answer, and the two ways it calls them.

A tool is a name, a description, a JSON Schema for its arguments, and a
handler that turns the arguments into a text result for the model. A
ToolSet is the tools one agent call is offered. reason() (loops/reasoning.py) runs
the loop: the model asks for a tool, the handler runs, the result goes
back, and the model goes on — until it gives its answer or the rounds run
out (ToolSet.max_rounds()).

A prompt mentions a tool only through the tool's own hint: reason() adds
the ToolSet's prompt_hints() to the task while it offers the tools, and
re-sends the task without them once it stops, so a model is never told to
call a tool it no longer has.

There are two ways a model asks, and the loop speaks both:

  * native — the OpenAI-style `tools` field of /v1/chat/completions. The
    model answers with structured tool_calls, the results go back as
    role "tool" messages. Models are trained on this, so it is the more
    reliable route, and the one any later tool (a computer algebra system,
    say) wants: many rounds, errors, long outputs. llama-server serves it
    when launched with --jinja.
  * json — the fallback for a server that refuses `tools`. The system
    prompt gains json_protocol(): reply with ONLY {"tool": NAME,
    "arguments": {...}} to call one. The result comes back as a user
    message. No agent's answer has a top-level "tool" key, which is what
    lets parse_json_call tell a call from an answer.

The tools themselves are plug-ins, each in its own package
(lookup_lemmas/, search_lemmas/, run_code/), and which of them an agent
call is offered is theirs to say (tools/__init__.py).

Every result lands in a prompt that shares one context window with the
answer still to come, so run() takes the room left for it, in characters
(reason() works it out before each call, keeping its reply reserve out of
it). A tool that knows its own structure trims to whole entries
(fit_entries: lookup_lemmas and search_lemmas return whole statements); any
other result is cut at a line boundary, with a marker saying so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# What a call gets when not even one entry of its result fits the room left:
# an instruction the model can act on, rather than a fragment it cannot.
NO_ROOM = (
    "error: there is no room left in the context window for this result. "
    "Make no more tool calls: give your answer now, with what you have."
)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]           # a JSON Schema object
    # handler(arguments) — or handler(arguments, budget) when takes_budget,
    # for a tool that trims its own result to whole entries.
    handler: Callable[..., str]
    takes_budget: bool = False
    # What the task prompt tells the model about this tool while it is
    # offered (ToolSet.prompt_hints): when to reach for it. "" for none.
    hint: str = ""
    # The rounds a call offered this tool may spend (ToolSet.max_rounds);
    # None for the ToolSet's default.
    max_rounds: Optional[int] = None
    # What a request that re-sends the task with the tool calls written in
    # (a fold, a compaction rescue) adds after them (ToolSet.fold_notes):
    # the state the calls led to. None for nothing.
    fold_note: Optional[Callable[[], str]] = None
    # Called when the agent call that was offered the tool ends
    # (ToolSet.close). None for nothing.
    on_close: Optional[Callable[[], None]] = None


def cut_to_budget(text: str, budget: Optional[int]) -> str:
    """text, cut at a line boundary to at most budget characters with a
    marker saying how much was left out — the generic trim, for a result
    whose tool does not trim itself. Unchanged when it fits or there is no
    budget; NO_ROOM when not even one line fits."""
    if budget is None or len(text) <= budget:
        return text
    marker = (
        f"\n[... cut here: the full result is {len(text):,} characters and "
        "the context window has room for only part of it. Room is running "
        "out: answer with what you have.]"
    )
    end = text.rfind("\n", 0, max(budget - len(marker), 0))
    if end <= 0:
        return NO_ROOM
    return text[:end] + marker


def fit_entries(entries: List[Any], render: Callable[[List[Any], int], str],
                budget: Optional[int]) -> str:
    """The trim of a tool that returns whole entries (lookup_lemmas,
    search_lemmas): render(entries, 0) when it fits the budget, or there is
    none; else render(the first k, how many were left out) for the largest
    k that fits; NO_ROOM when not even one does. Half a statement is worse
    than none — the model would build on a hypothesis list cut short."""
    text = render(entries, 0)
    if budget is None or len(text) <= budget:
        return text
    best = NO_ROOM
    for k in range(1, len(entries)):
        trial = render(entries[:k], len(entries) - k)
        if len(trial) > budget:
            break
        best = trial
    return best


@dataclass
class ToolSet:
    tools: List[Tool] = field(default_factory=list)
    # How many tool rounds one agent call may spend before it must answer,
    # for a tool that does not set its own (tools.max_rounds). A round is
    # one reply from the model and the results of the calls in it; every
    # round's results stay in the prompt, so the bound is what keeps the
    # prompt's growth bounded too.
    default_rounds: int = 3

    def __bool__(self) -> bool:
        return bool(self.tools)

    def names(self) -> List[str]:
        return [t.name for t in self.tools]

    def max_rounds(self) -> int:
        """The rounds one agent call offered these tools may spend: the
        most any of them asks for, -1 (no limit) when one asks for that."""
        rounds = [t.max_rounds or self.default_rounds for t in self.tools]
        if not rounds:
            return self.default_rounds
        return -1 if min(rounds) < 0 else max(rounds)

    def prompt_hints(self) -> str:
        """The block reason() adds to the task while the tools are offered:
        each tool's hint. "" when none has one."""
        hints = [f"- {t.name}: {t.hint}" for t in self.tools if t.hint]
        if not hints:
            return ""
        return "Tools you can call before you answer:\n" + "\n".join(hints)

    def fold_notes(self) -> str:
        """What the tools add to a task re-sent with their calls written
        in: each one's fold_note. "" when none has anything to add."""
        notes = [n for n in (t.fold_note() for t in self.tools if t.fold_note) if n]
        return "\n\n".join(notes)

    def close(self) -> None:
        """The agent call is over: each tool's on_close."""
        for t in self.tools:
            if t.on_close:
                t.on_close()

    def specs(self) -> List[Dict[str, Any]]:
        """The OpenAI-style `tools` field, for a native call."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.tools
        ]

    def run(self, name: str, arguments: Any,
            budget: Optional[int] = None) -> str:
        """The result of one call, as the text the model reads. A bad call
        (unknown tool, arguments that are not an object, a handler that
        raises) is answered with an error the model can act on, never
        raised: a tool call is the model's move, and a wrong one is its
        to correct.

        budget, when given, is the most characters the result may take: the
        room the prompt has left before it would cut into the reply's
        reserve. A tool that takes_budget trims itself to whole entries;
        every result is then held to the budget by cut_to_budget, which
        leaves one that fits untouched."""
        if budget is not None and budget <= 0:
            return NO_ROOM
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError as e:
                return f"error: the arguments are not valid JSON ({e})"
        if not isinstance(arguments, dict):
            return "error: the arguments must be a JSON object"
        for t in self.tools:
            if t.name == name:
                try:
                    result = (t.handler(arguments, budget) if t.takes_budget
                              else t.handler(arguments))
                except Exception as e:  # the model's call, the model's error
                    result = f"error: {name} failed: {e}"
                return cut_to_budget(result, budget)
        return f"error: there is no tool {name!r}; the tools are {self.names()}"

    def json_protocol(self) -> str:
        """The fallback protocol, appended to the system prompt when the
        server takes no native tools."""
        lines = [
            "You have tools. To call one, reply with ONLY a JSON object of "
            'the form {"tool": "<name>", "arguments": {...}} and nothing '
            "else; the result comes back in the next message. "
            + ("Call as many times as you need"
               if self.max_rounds() < 0
               else f"Call at most {self.max_rounds()} times")
            + ", then give your answer in the format asked for above. "
            "The tools:",
        ]
        for t in self.tools:
            lines.append(
                f"- {t.name}: {t.description} Arguments (JSON Schema): "
                f"{json.dumps(t.parameters)}"
            )
        return "\n".join(lines)

    def parse_json_call(self, content: str) -> Optional[Tuple[str, Any]]:
        """(name, arguments) when content is a fallback-protocol call to one
        of these tools, else None (it is the answer)."""
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        if not text.startswith("{"):
            return None
        try:
            obj = json.loads(text)
        except ValueError:
            return None
        if not isinstance(obj, dict) or obj.get("tool") not in self.names():
            return None
        return obj["tool"], obj.get("arguments", {})
