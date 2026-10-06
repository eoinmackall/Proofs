"""Tools: things an agent may call mid-answer, and the two ways it calls them.

A tool is a name, a description, a JSON Schema for its arguments, and a
handler that turns the arguments into a text result for the model. A
ToolSet is the tools one agent call is offered. reason() in main.py runs
the loop: the model asks for a tool, the handler runs, the result goes
back, and the model goes on — until it gives its answer or the rounds run
out (MAX_TOOL_ROUNDS).

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

The first tool is lookup_lemmas: the lemma window (main.lemma_window)
shows the agents the most recently proved lemmas in full and lists the
rest by id only; this is how an agent reads one of the rest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

# How many tool rounds one agent call may spend before it must answer. A
# round is one reply from the model and the results of the calls in it;
# every round's results stay in the prompt, so the bound is what keeps the
# prompt's growth bounded too.
MAX_TOOL_ROUNDS = 3

# How many lemmas one lookup_lemmas call may return, for the same reason.
MAX_LOOKUP = 10


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]           # a JSON Schema object
    handler: Callable[[Dict[str, Any]], str]


@dataclass
class ToolSet:
    tools: List[Tool] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.tools)

    def names(self) -> List[str]:
        return [t.name for t in self.tools]

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

    def run(self, name: str, arguments: Any) -> str:
        """The result of one call, as the text the model reads. A bad call
        (unknown tool, arguments that are not an object, a handler that
        raises) is answered with an error the model can act on, never
        raised: a tool call is the model's move, and a wrong one is its
        to correct."""
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
                    return t.handler(arguments)
                except Exception as e:  # the model's call, the model's error
                    return f"error: {name} failed: {e}"
        return f"error: there is no tool {name!r}; the tools are {self.names()}"

    def json_protocol(self) -> str:
        """The fallback protocol, appended to the system prompt when the
        server takes no native tools."""
        lines = [
            "You have tools. To call one, reply with ONLY a JSON object of "
            'the form {"tool": "<name>", "arguments": {...}} and nothing '
            "else; the result comes back in the next message. Call at most "
            f"{MAX_TOOL_ROUNDS} times, then give your answer in the format "
            "asked for above. The tools:",
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


Pair = Tuple[str, str]


def lemma_lookup(
    lemmas: Mapping[Pair, Mapping[str, Any]],
    suspended: Optional[Set[Pair]] = None,
) -> Tool:
    """lookup_lemmas over the complete DAG's lemmas (load_dag()'s mapping,
    keyed by (user_id, lemma_id)). It returns what the lemma window shows
    for a lemma in full — the statement and the citations, not the proof —
    and refuses a suspended lemma, as the window does: a lemma no agent may
    build on must not come back in through a lookup.

    user_id may be left out, in which case every user's lemma of that
    lemma_id is returned (a lemma_id is unique only within one user's
    file)."""
    suspended = suspended or set()

    def handler(args: Dict[str, Any]) -> str:
        wanted = args.get("lemmas")
        if not isinstance(wanted, list) or not wanted:
            return 'error: pass "lemmas", a non-empty list of {"user_id", "lemma_id"} objects'
        found: List[Dict[str, Any]] = []
        for entry in wanted[:MAX_LOOKUP]:
            if isinstance(entry, str):
                entry = {"lemma_id": entry}
            if not isinstance(entry, dict):
                continue
            lid = str(entry.get("lemma_id") or "").strip()
            uid = str(entry.get("user_id") or "").strip()
            pairs = (
                [(uid, lid)] if uid
                else sorted(p for p in lemmas if p[1] == lid)
            )
            hit = False
            for pair in pairs:
                if pair not in lemmas:
                    continue
                hit = True
                if pair in suspended:
                    found.append({
                        "user_id": pair[0], "lemma_id": pair[1],
                        "error": "suspended: not available to build on",
                    })
                    continue
                node = lemmas[pair]
                found.append({
                    "user_id": pair[0],
                    "lemma_id": pair[1],
                    "statement": str(node.get("statement", "")),
                    "cited_lemmas": [
                        {"user_id": d[0], "lemma_id": d[1]}
                        if isinstance(d, tuple) else d
                        for d in node.get("cited_lemmas", [])
                    ],
                    "cited_references": list(node.get("cited_references", [])),
                })
            if not hit:
                found.append({
                    "user_id": uid, "lemma_id": lid,
                    "error": "no such proved lemma",
                })
        note = (
            f" (only the first {MAX_LOOKUP} were looked up)"
            if len(wanted) > MAX_LOOKUP else ""
        )
        return json.dumps({"lemmas": found}, indent=2) + note

    return Tool(
        name="lookup_lemmas",
        description=(
            "Read the statements of proved lemmas listed in "
            "other_proved_lemmas by id only. Give each as its user_id and "
            "lemma_id; leave user_id out to get every user's lemma of that "
            f"lemma_id. At most {MAX_LOOKUP} lemmas per call."
        ),
        parameters={
            "type": "object",
            "properties": {
                "lemmas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "lemma_id": {"type": "string"},
                        },
                        "required": ["lemma_id"],
                    },
                },
            },
            "required": ["lemmas"],
        },
        handler=handler,
    )
