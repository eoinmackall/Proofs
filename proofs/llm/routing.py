"""Model routing: which model serves each agent role.

A command may talk to several models. Which one a call goes to is decided
by its role — the agent making it: planner, prover, verifier_1, ... —
through a routing table built from two keys of .proofs-local.json:

    "llm.models": {
        "local":  {"host": "http://localhost:8080/v1", "model": "qwen3-32b"},
        "strong": {"host": "https://api.openai.com/v1", "model": "gpt-5",
                   "api_key_env": "OPENAI_API_KEY", "ctx_budget": 200000}
    },
    "llm.routes": {
        "default":    "local",
        "prover":     "strong",
        "verifier_*": "strong",
        "extract":    "local"
    }

A model entry names a server the way the backend flags do: host and model
(host optional for backend anthropic, which defaults to
https://api.anthropic.com), backend (auto, llamacpp, openai, or anthropic
for the Anthropic Messages API; default auto), ctx_budget (the most one
call may take, prompt plus generation; default DEFAULT_CTX_BUDGET, capped
by the server's own window where the probe reads one), temperature (sent with every call to it; by
default none is sent, and the server's own applies; the Anthropic API
takes none), effort (how hard a reasoning model thinks, where the API
takes it: EFFORTS lists each backend's levels; by default none is sent)
and api_key_env, the name of the environment variable that holds its API
key. A key itself is never read
from a file: a gitignored file is still one `git add -f`, backup or
tarball away from being shared, so a file that holds one is refused.

A route maps a role, or a glob over the roles (verifier_*), to a model's
name. "default" serves every role nothing else routes.

A blank value ("" or null) is unset: a blank field of a model entry is
left out, a model entry with nothing but its backend filled in defines no
model (and routing a role to it is an error), and a blank route routes
nothing. proofs init writes the file that way, every route and two model
entries present but blank (settings.ROUTING_TEMPLATE), so the file works
as it stands and is filled in rather than written from scratch. Two stages can be
routed apart from their role: "extract", the schema extraction that copies
a role's prose into JSON, and "compact", the summarising pass of a call
that hit the context wall; unrouted, each goes to its role's model.

The backend flags (--host, --model, --api-key, --backend, --ctx-budget,
--temperature, --effort) describe one more model, and make it the default route; --route ROLE=NAME
routes one role (or glob) for one command. A command line that worked
before routing works the same, and a file that routes a default needs no
backend flags at all.

The layers, lowest first; each route and each model name is resolved on
its own, and the last layer that sets it wins:

    <workspace>/.proofs-local.json
    conjectures/NAME/.proofs-local.json
    the backend flags, then --route

Within one layer an exact role beats a glob; across layers the later layer
wins, whichever form either takes (a conjecture's "verifier_*" overrides
the workspace's "verifier_1"). "default" is layered the same way: a
conjecture's default replaces the workspace's, but no default reaches a
role that some layer routes. A model entry is replaced whole, never merged
field by field.

With --with-fallback, a call whose model cannot serve it — its quota or
credit used up, or no answer after the loops' retries — is sent again to
the fallback model: the "fallback" route (a route key like "default",
layered the same way, and --route fallback=NAME), else the default
route's model. It is per call, never sticky: the next call goes to its
own model first, so a rate limit that has passed costs nothing. A call
already on the fallback model has nothing to fall back to, and keeps the
retry behaviour it has without the flag.

Some roles must share a model (Models.build's same_model groups): the
loops pass every verifier, since a certificate records one model for the
whole check. A group's members follow each other: when the members some
layer routes agree, the rest take their model rather than the default, so
"verifier_*" also routes coding_verifier, and one route sets every
verifier. Members routed to different models are refused.

Nothing here knows the loops: the roles are passed in, and the state the
loops learn about a model as they go (whether it takes native tool calls,
whether schema mode works) lives on its Model, one per configured model,
shared by every role routed to it.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import DEFAULT_CTX_BUDGET

DEFAULT_ROUTE = "default"
FALLBACK_ROUTE = "fallback"
STAGES = ("extract", "compact")
# The name of the model the backend flags describe: not a valid name in a
# file (_NAME_RE), so it never collides with one.
FLAG_MODEL = "--model"
BACKENDS = ("auto", "llamacpp", "openai", "anthropic")
# The host a backend defaults to, when it has one.
DEFAULT_HOSTS = {"anthropic": "https://api.anthropic.com"}
# The fields a file's model entry may hold.
_FIELDS = ("host", "model", "backend", "ctx_budget", "temperature", "effort",
           "api_key_env")
# The effort levels each API takes: the Anthropic API's output_config.effort
# (which levels a model takes is probed, AnthropicBackend.effort_error), and
# the OpenAI-compatible reasoning_effort, whose levels vary by provider and
# model and cannot be probed — a level the server refuses fails its calls.
EFFORTS = {
    "anthropic": ("low", "medium", "high", "xhigh", "max"),
    "openai": ("none", "minimal", "low", "medium", "high", "xhigh"),
}


def check_effort(backend: str, effort: Any, where: str) -> Optional[str]:
    """effort if backend's API takes it, else RoutingError naming the
    levels it does take. None passes through."""
    if effort is None:
        return None
    levels = EFFORTS["anthropic" if backend == "anthropic" else "openai"]
    if not isinstance(effort, str) or effort not in levels:
        raise RoutingError(
            f"{where}: effort must be one of {', '.join(levels)} for backend "
            f"{backend}, not {effort!r}")
    return effort
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_GLOB_CHARS = frozenset("*?[")


def _blank(value: Any) -> bool:
    """A value left unfilled: null, or a string of nothing but spaces."""
    return value is None or (isinstance(value, str) and not value.strip())


def unfilled(raw: Any) -> bool:
    """An llm.models entry nothing but its backend has been filled in for,
    like the ones proofs init writes: it defines no model."""
    return isinstance(raw, dict) and all(
        _blank(v) for k, v in raw.items() if k != "backend")


class RoutingError(ValueError):
    """The routing config is unusable: a malformed entry, a route to a
    model that is not defined, a role no route reaches."""


@dataclass(frozen=True)
class ModelSpec:
    """One configured model: where it is served and how to talk to it."""

    host: str
    model: str
    backend: str = "auto"
    ctx_budget: Optional[int] = None
    # Sent with every call when set; None leaves it to the server.
    temperature: Optional[float] = None
    # The reasoning effort (EFFORTS), sent the same way.
    effort: Optional[str] = None
    api_key_env: Optional[str] = None
    # From --api-key only: a file never holds a key.
    api_key: Optional[str] = field(default=None, repr=False)
    # Where it was defined, for messages: a file, or "flags".
    source: str = ""

    def key(self, name: str) -> Optional[str]:
        """The API key to send, None for none. A model that names an
        environment variable needs it set: a missing key would fail every
        call, so it is an error before the first one."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if not value:
                raise RoutingError(
                    f"model {name!r} ({self.source}) takes its API key from "
                    f"${self.api_key_env}, which is not set"
                )
            return value
        return None


def check_entry(name: str, raw: Any, source: str) -> Dict[str, Any]:
    """The structural checks every llm.models entry gets, blank or not: its
    name, its shape, its field names and its backend. Returns raw without
    its blank fields ("" or null, as proofs init writes them: unset);
    RoutingError for anything malformed."""
    where = f"{source}: llm.models.{name}"
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise RoutingError(
            f"{source}: {name!r} is not a model name: start with a letter "
            f"or digit, then letters, digits, dots, underscores or hyphens"
        )
    if not isinstance(raw, dict):
        raise RoutingError(f"{where} must be an object (host, model, ...)")
    if "api_key" in raw:
        raise RoutingError(
            f"{where} holds an api_key. A key is never read from a file: put "
            f"it in an environment variable and name the variable with "
            f"api_key_env"
        )
    if "num_ctx" in raw:
        raise RoutingError(
            f"{where}: num_ctx is now ctx_budget (the most one call may "
            f"take, prompt plus generation); rename the field")
    unknown = sorted(set(raw) - set(_FIELDS))
    if unknown:
        raise RoutingError(
            f"{where}: unknown field(s) {', '.join(unknown)}; the fields are "
            f"{' '.join(_FIELDS)}"
        )
    raw = {k: v for k, v in raw.items() if not _blank(v)}
    backend = raw.get("backend", "auto")
    if backend not in BACKENDS:
        raise RoutingError(
            f"{where}: backend must be one of {', '.join(BACKENDS)}, "
            f"not {backend!r}"
        )
    return raw


def parse_model(name: str, raw: Any, source: str) -> ModelSpec:
    """The ModelSpec a file's llm.models entry describes; RoutingError for
    anything malformed."""
    where = f"{source}: llm.models.{name}"
    raw = check_entry(name, raw, source)
    backend = raw.get("backend", "auto")
    host = raw.get("host", DEFAULT_HOSTS.get(backend))
    for required, value in (("host", host), ("model", raw.get("model"))):
        if not isinstance(value, str) or not value.strip():
            raise RoutingError(f"{where} needs a {required}")
    ctx_budget = raw.get("ctx_budget")
    if ctx_budget is not None and (isinstance(ctx_budget, bool)
                                or not isinstance(ctx_budget, int) or ctx_budget < 1):
        raise RoutingError(
            f"{where}: ctx_budget must be a positive integer, not {ctx_budget!r}")
    temperature = raw.get("temperature")
    if temperature is not None and (isinstance(temperature, bool)
                                    or not isinstance(temperature, (int, float))
                                    or temperature < 0):
        raise RoutingError(
            f"{where}: temperature must be a number of at least 0, "
            f"not {temperature!r}")
    if temperature is not None and backend == "anthropic":
        raise RoutingError(
            f"{where}: the Anthropic API takes no temperature (it refuses "
            f"one on every model); remove it")
    effort = check_effort(backend, raw.get("effort"), where)
    env = raw.get("api_key_env")
    if env is not None and (not isinstance(env, str) or not _ENV_RE.match(env)):
        raise RoutingError(
            f"{where}: api_key_env must name an environment variable, "
            f"not {env!r}")
    return ModelSpec(host=host.strip(), model=raw["model"].strip(),
                     backend=backend, ctx_budget=ctx_budget,
                     temperature=(float(temperature) if temperature is not None
                                  else None),
                     effort=effort,
                     api_key_env=env, source=source)


@dataclass(frozen=True)
class Flags:
    """The backend flags of one command (cli/flags.py, model_flags)."""

    host: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = field(default=None, repr=False)
    backend: Optional[str] = None
    ctx_budget: Optional[int] = None
    temperature: Optional[float] = None
    effort: Optional[str] = None
    # --route ROLE=NAME, as given.
    routes: Tuple[str, ...] = ()
    # --with-fallback: fail a call over to the fallback model.
    with_fallback: bool = False

    def spec(self) -> Optional[ModelSpec]:
        """The model --host and --model describe, None when neither is
        given. The other flags describe that model, so they need it."""
        if self.host is None and self.model is None:
            given = [flag for flag, value in (
                ("--api-key", self.api_key), ("--backend", self.backend),
                ("--ctx-budget", self.ctx_budget),
                ("--temperature", self.temperature),
                ("--effort", self.effort)) if value is not None]
            if given:
                raise RoutingError(
                    f"{', '.join(given)} describe{'s' if len(given) == 1 else ''} "
                    f"the model --host and --model name: pass those too, or "
                    f"set it on the model's entry in llm.models"
                )
            return None
        host = self.host or DEFAULT_HOSTS.get(self.backend or "")
        if host is None or self.model is None:
            raise RoutingError("pass --host and --model together")
        if self.temperature is not None and self.backend == "anthropic":
            raise RoutingError(
                "--temperature: the Anthropic API takes no temperature (it "
                "refuses one on every model)")
        return ModelSpec(host=host, model=self.model,
                         backend=self.backend or "auto", ctx_budget=self.ctx_budget,
                         temperature=self.temperature,
                         effort=check_effort(self.backend or "auto",
                                             self.effort, "--effort"),
                         api_key=self.api_key or None, source="flags")

    def route_layer(self) -> Dict[str, str]:
        """--route's ROLE=NAME pairs as a routes layer."""
        layer: Dict[str, str] = {}
        for text in self.routes:
            role, sep, name = (part.strip() for part in text.partition("="))
            if not sep or not role or not name:
                raise RoutingError(
                    f"--route {text!r}: expected ROLE=NAME, e.g. prover=strong")
            layer[role] = name
        return layer


@dataclass(eq=False)
class Model:
    """One configured model, connected: its backend, what the probe found,
    the context window it runs with, and what the run has learned about it
    — properties of the model, learned once and kept for the process,
    shared by every role routed to it."""

    name: str
    spec: ModelSpec
    # The llm.Backend that owns the HTTP conversation, and its probe.
    backend: Any = None
    profile: Any = None
    # The context budget in tokens: the most one call may take, prompt
    # plus generation (llm.connect settles it).
    ctx_budget: int = DEFAULT_CTX_BUDGET
    # Native tool calls: None until the server refuses one, then False.
    native_tools_ok: Optional[bool] = None
    # Schema-constrained extraction proved incompatible with the model's
    # default thinking (loops/reasoning.py, extract()).
    schema_mode_broken: bool = False
    # The model needs its reasoning trace sent back on tool-call turns
    # (loops/reasoning.py, reason()).
    echo_reasoning: bool = False
    # The server refused extract()'s greedy sampling fields once: they are
    # not sent to it again.
    greedy_refused: bool = False

    @property
    def requested(self) -> str:
        """The model name requests carry: the entry's model, or --model."""
        return self.spec.model

    def recorded(self) -> str:
        """The name certificates and refutations record (the backend's
        recorded_model()), the requested one before there is a backend."""
        return self.backend.recorded_model() if self.backend is not None else self.requested

    def recorded_known(self) -> bool:
        """Whether recorded() is read off the server yet."""
        return self.backend is not None and self.backend.recorded_model_known()

    def label(self) -> str:
        """How messages name it: "strong (gpt-5)", or "--model gpt-5"."""
        if self.name == FLAG_MODEL:
            return f"--model {self.requested}"
        return f"{self.name} ({self.requested})"

    def ctx_budget_fix(self, tokens: int) -> str:
        """The instruction that sets this model's budget to tokens."""
        if self.name == FLAG_MODEL:
            return f"re-run with --ctx-budget {tokens}"
        return (f"set ctx_budget to {tokens} on llm.models.{self.name} in "
                f"{self.spec.source}")


class Models:
    """The models a command talks to, and the role each serves: built (and
    checked, offline) by build(), connected by connect()."""

    def __init__(self, roles: Mapping[str, Model],
                 stages: Mapping[str, Model],
                 fallback: Optional[Model] = None) -> None:
        self._roles = dict(roles)
        self._stages = dict(stages)
        # The roles the command calls (connect()); every role until then.
        self._used: Tuple[str, ...] = tuple(self._roles)
        #: The model a failed call goes to (--with-fallback); None without
        #: the flag.
        self.fallback = fallback

    def for_role(self, role: str) -> Model:
        try:
            m = self._roles[role]
        except KeyError:
            raise KeyError(f"no model is routed for role {role!r}; the roles "
                           f"are {', '.join(self._roles)}") from None
        if role not in self._used:
            # A command connects only the roles it calls: reaching another
            # is a bug in the command's list, not a configuration problem.
            raise RuntimeError(f"role {role!r} was not connected for this "
                               f"command (it connects {', '.join(self._used)})")
        return m

    def for_stage(self, stage: str, role: str) -> Model:
        """The model a stage of role's call goes to: the stage's own route,
        else the role's model."""
        return self._stages.get(stage) or self.for_role(role)

    def stage_route(self, stage: str) -> Optional[Model]:
        """The stage's own route, None when it follows its role."""
        return self._stages.get(stage)

    def distinct(self) -> List[Model]:
        """Every model the command's roles use, each once, in the order of
        first use: their routes, the stages' routes (the roles' calls
        extract and compact through them) and the fallback."""
        seen: Dict[int, Model] = {}
        extra = (self.fallback,) if self.fallback is not None else ()
        used = [self._roles[r] for r in self._used]
        for m in (*used, *self._stages.values(), *extra):
            seen.setdefault(id(m), m)
        return list(seen.values())

    def routed(self) -> bool:
        """Whether more than one model serves the command."""
        return len(self.distinct()) > 1

    def connect(self, timeout: int,
                roles: Optional[Sequence[str]] = None) -> List[str]:
        """Make and probe the backend of each model the command uses — the
        roles it calls (every role when None), with their stages and the
        fallback — settling its window: a model only an uncalled role uses
        is neither probed nor needs its key. The startup notes
        (llm.describe), each naming its model when there is more than one. llm.ProbeError for a model the probe found
        unservable (unreachable, wrong base URL, key, unknown model), prefixed
        with the model; RoutingError for a key's variable not set."""
        from . import ProbeError, connect

        if roles is not None:
            self._used = tuple(r for r in self._roles if r in set(roles))
        many = self.routed()
        notes: List[str] = []
        for m in self.distinct():
            key = m.spec.key(m.name)
            try:
                c = connect(m.spec.model, m.spec.host, timeout, key,
                            m.spec.backend, m.spec.ctx_budget)
            except ProbeError as e:
                raise ProbeError(f"{m.label()}: {e}" if many else str(e)) from None
            m.backend, m.profile, m.ctx_budget = c.backend, c.profile, c.ctx_budget
            m.echo_reasoning = c.backend.echoes_reasoning
            if m.spec.effort is not None:
                # A level the probe shows the model does not take would
                # fail every call: refused here, before the first.
                refused = c.backend.effort_error(m.spec.effort)
                if refused:
                    raise RoutingError(f"{m.label()}: {refused}")
            if c.notes:
                notes.append(f"{m.label()}: {c.notes}" if many else c.notes)
        return notes

    def describe(self) -> str:
        """Which model serves which roles, a line per model, for the log."""
        lines = ["models:"]
        for m in self.distinct():
            roles = [r for r in self._used if self._roles[r] is m]
            roles += [f"{s} stage" for s, sm in self._stages.items() if sm is m]
            if m is self.fallback:
                roles.append("fallback")
            lines.append(
                f"  {m.label()} at {m.spec.host}, {m.ctx_budget:,}-token budget: "
                f"{', '.join(roles)}"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        """The models the command records, for a one-line header."""
        return ", ".join(dict.fromkeys(m.recorded() for m in self.distinct()))

    @classmethod
    def build(
        cls,
        model_layers: Sequence[Tuple[str, Any]],
        route_layers: Sequence[Tuple[str, Any]],
        flags: Flags,
        roles: Sequence[str],
        same_model: Sequence[Tuple[Sequence[str], str]] = (),
    ) -> "Models":
        """The routing table for roles, from the files' llm.models and
        llm.routes ((source, value) pairs, lowest layer first) and the
        backend flags. same_model is (roles, why) pairs: each group must
        resolve to one model, why saying what depends on that. Offline:
        nothing is connected. RoutingError for anything unusable."""
        specs: Dict[str, ModelSpec] = {}
        blank_models: Dict[str, str] = {}   # name -> source, entries left blank
        for source, layer in model_layers:
            if not isinstance(layer, dict):
                raise RoutingError(
                    f"{source}: llm.models must be an object mapping a name "
                    f"to a model")
            for name, raw in layer.items():
                if unfilled(raw):
                    # A template entry no one has filled in: it defines no
                    # model, and never replaces a lower layer's entry. It is
                    # still checked, so a misspelt field or backend is not
                    # mistaken for a placeholder.
                    check_entry(name, raw, source)
                    blank_models.setdefault(name, source)
                    continue
                specs[name] = parse_model(name, raw, source)
        flag_spec = flags.spec()
        layers = list(route_layers)
        if flag_spec is not None:
            specs[FLAG_MODEL] = flag_spec
            layers.append(("--host/--model", {DEFAULT_ROUTE: FLAG_MODEL}))
        cli_routes = flags.route_layer()
        if cli_routes:
            layers.append(("--route", cli_routes))

        # role or stage -> (model name, source); default the same, or None.
        routes: Dict[str, Tuple[str, str]] = {}
        default: Optional[Tuple[str, str]] = None
        fallback: Optional[Tuple[str, str]] = None
        for source, layer in layers:
            if not isinstance(layer, dict):
                raise RoutingError(
                    f"{source}: llm.routes must be an object mapping a role "
                    f"to a model's name")
            exact: Dict[str, str] = {}
            globbed: Dict[str, Tuple[str, str]] = {}   # role -> (glob, name)
            for key, target in layer.items():
                # A blank route ("" or null, as proofs init writes every
                # one) routes nothing; its key must still be a route.
                blank = _blank(target)
                if not blank and not isinstance(target, str):
                    raise RoutingError(
                        f"{source}: llm.routes.{key} must name a model")
                if key == DEFAULT_ROUTE:
                    if not blank:
                        default = (target, source)
                elif key == FALLBACK_ROUTE:
                    if not blank:
                        fallback = (target, source)
                elif key in roles or key in STAGES:
                    if not blank:
                        exact[key] = target
                elif _GLOB_CHARS & set(key):
                    matched = [r for r in roles if fnmatch.fnmatchcase(r, key)]
                    if not matched:
                        raise RoutingError(
                            f"{source}: route {key!r} matches no role; the "
                            f"roles are {', '.join(roles)}")
                    if blank:
                        continue
                    for r in matched:
                        if r in globbed and globbed[r][1] != target:
                            raise RoutingError(
                                f"{source}: routes {globbed[r][0]!r} and "
                                f"{key!r} both match {r} but name different "
                                f"models; route {r} itself to say which")
                        globbed[r] = (key, target)
                else:
                    raise RoutingError(
                        f"{source}: {key!r} is not a route. The routes are "
                        f"{DEFAULT_ROUTE}, {FALLBACK_ROUTE}, the roles "
                        f"({', '.join(roles)}), "
                        f"the stages ({', '.join(STAGES)}), or a glob over "
                        f"the roles (verifier_*)")
            for r, (_glob, target) in globbed.items():
                exact.setdefault(r, target)
            for key, target in exact.items():
                routes[key] = (target, source)

        # A role that must share a model with others follows them: when the
        # routed members of a same_model group agree, the group's unrouted
        # members take that model instead of the default. So "verifier_*"
        # (or "verifier_1" alone) routes coding_verifier too, and one route
        # sets the whole group. Members routed apart are refused below.
        for group, _why in same_model:
            routed = [routes[r] for r in group if r in routes]
            if routed and len({target for target, _s in routed}) == 1:
                target, source = routed[0]
                for r in group:
                    if r in roles and r not in routes:
                        routes[r] = (target, source)

        unrouted = [r for r in roles if r not in routes]
        if unrouted and default is None:
            which = ("no model" if len(unrouted) == len(roles)
                     else f"no model for {', '.join(unrouted)}")
            raise RoutingError(
                f"{which}: pass --host and --model, or route a default in "
                f".proofs-local.json (llm.routes.default)")
        resolved = {r: routes.get(r, default) for r in roles}
        stage_routes = {s: routes[s] for s in STAGES if s in routes}
        checked = [*resolved.values(), *stage_routes.values()]
        if default is not None:
            checked.append(default)
        if fallback is not None:
            checked.append(fallback)
        if flags.with_fallback and fallback is None:
            fallback = default
            if fallback is None:
                raise RoutingError(
                    "--with-fallback needs a model to fall back to: route "
                    "one with llm.routes.fallback (or --route "
                    "fallback=NAME), or a default")
        for target, source in checked:
            if target not in specs and target in blank_models:
                raise RoutingError(
                    f"{source}: route to {target!r}, but llm.models.{target} "
                    f"({blank_models[target]}) is still blank: fill in its "
                    f"model, and its host unless its backend is anthropic")
            if target not in specs:
                defined = ", ".join(n for n in specs if n != FLAG_MODEL)
                raise RoutingError(
                    f"{source}: route to {target!r}, which no llm.models "
                    f"entry defines (defined: {defined or 'none'})")
        for group, why in same_model:
            names = {resolved[r][0] for r in group if r in resolved}
            if len(names) > 1:
                pairs = ", ".join(f"{r} -> {resolved[r][0]}" for r in group
                                  if r in resolved)
                raise RoutingError(
                    f"{', '.join(group)} must all route to one model "
                    f"({why}); they route {pairs}")

        models: Dict[str, Model] = {}

        def model(name: str) -> Model:
            if name not in models:
                models[name] = Model(name=name, spec=specs[name])
            return models[name]

        return cls({r: model(t) for r, (t, _s) in resolved.items()},
                   {s: model(t) for s, (t, _s) in stage_routes.items()},
                   model(fallback[0]) if flags.with_fallback and fallback
                   else None)
