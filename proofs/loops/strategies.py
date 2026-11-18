"""The lemma window's strategies: which proved lemmas an agent is shown in
full (--strategy). Every other proved lemma is found with the tools
(search_lemmas, lookup_lemmas); the prompt says only how many there are.

    recent      the --lemmas-all newest lemmas, anyone's, and the
                --my-lemmas newest of the current user's own: half of
                --window-size (rounded down) each unless they are given
    random      a sample of the lemmas, weighted by depth (the longest
                citation chain beneath a lemma): weight (depth + 1) to the
                power beta / (1 - beta), so beta 0 is uniform and beta 1
                the deepest lemmas
    depth       sinks (lemmas no unsuspended lemma cites: nobody has built
                on them yet) by depth, the same weights with the dial the
                other way round: jitter 0 is the deepest sinks, jitter 1
                any sink, uniformly
    breadth     sinks whose ancestries overlap least: from the deepest
                sink, each next the one least like every sink chosen so
                far (citations.Ancestry estimates the overlap); at most
                half of them sinks that cite nothing, whose empty
                ancestries are unlike everything
    relevance   what a search (tools/search_lemmas) finds for the agent's
                task: the lemma being proved, for the prover and the
                reviser; the latest plan for the planner, the conjecture
                before there is one
    popular     the lemmas the most unsuspended lemmas cite

Every other strategy shows --window-size lemmas; ties are broken by
recency (proved_at, then id). Suspended lemmas are never chosen.

A window is chosen when a planning agent starts (for_planner): the parallel
lemma generator is then shown the planner's (for_generator), and so are the
prover and the reviser of a serial run (Context.run.window_from_planner).
A parallel loop's prover, or a repair's, chooses its own when it starts
(for_prover), and its reviser is shown the same one. Between those points
the window holds, whatever the DAG does: a lemma suspended since is left
out when the window is shown, a lemma proved since waits for the next
choice. relevance is the one strategy a serial prover chooses for itself,
since its query is the lemma being proved; it keeps it while it works on
that lemma.

A Window is its strategy's choice in order of preference, in parts (two
for recent, one otherwise), so that _fit_prompt (prompts.py) can halve
each part to fit a prompt and keep the preferred lemmas.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.citations import Ancestry
from ..core.dag import owner_of

# Without numpy, breadth compares this many sinks, drawn at random (with
# the deepest), rather than every one.
BREADTH_POOL = 2000


@dataclass(frozen=True)
class Window:
    """A strategy's choice: its parts, each in order of preference, and the
    query it was chosen for (relevance; "" otherwise)."""

    parts: Tuple[Tuple[str, ...], ...]
    query: str = ""

    def sizes(self) -> Tuple[int, ...]:
        return tuple(len(part) for part in self.parts)

    def shown(self, dag: Any, suspended: Collection[str],
              sizes: Optional[Sequence[int]] = None) -> List[str]:
        """The lemmas shown: each part cut to its size (sizes, or all of
        it), without repeats, without a lemma suspended since or not in
        dag."""
        out: List[str] = []
        seen = set()
        for part, n in zip(self.parts, sizes if sizes is not None else self.sizes()):
            for lid in part[:n]:
                if lid not in seen and lid in dag.lemmas and lid not in suspended:
                    seen.add(lid)
                    out.append(lid)
        return out


def plan_query(conjecture: str, plan: Optional[Mapping[str, Any]]) -> str:
    """relevance's query for a planning agent: the latest plan — its
    summary and its candidates (serial) or priorities (parallel) — or the
    conjecture before there is one."""
    if plan:
        texts = [str(plan.get("plan_summary") or "")]
        texts += [str(c.get("statement") or "") for c in plan.get("candidate_lemmas") or ()
                  if isinstance(c, Mapping)]
        texts += [str(p) for p in plan.get("priorities") or ()]
        text = "\n".join(t for t in texts if t.strip())
        if text:
            return text
    return conjecture


def for_planner(ctx, dag: Any, suspended: Collection[str], query: str) -> Window:
    """A planning agent starts: choose its window, the one the run's other
    agents are shown until the next planning call."""
    window = choose(ctx, dag, suspended, query)
    ctx.run.planning_window = window
    ctx.loop.window = window
    return window


def for_generator(ctx, dag: Any, suspended: Collection[str], query: str) -> Window:
    """The parallel lemma generator is shown the planner's window (chosen
    here, for both, when the planner has not chosen one yet)."""
    window = ctx.run.planning_window
    if window is None:
        window = ctx.run.planning_window = choose(ctx, dag, suspended, query)
    ctx.loop.window = window
    return window


def for_prover(ctx, dag: Any, suspended: Collection[str], statement: str) -> Window:
    """A prover starts on statement: the planner's window in a serial run
    (its own for relevance, kept while it stays on statement), its own
    otherwise. The reviser after it is shown the same."""
    if ctx.run.window_from_planner:
        if ctx.settings.strategy == "relevance":
            window = ctx.loop.window
            if window is None or window.query != statement:
                window = choose(ctx, dag, suspended, statement)
        else:
            window = ctx.run.planning_window
            if window is None:      # a resumed run, past its planning step
                window = ctx.run.planning_window = choose(ctx, dag, suspended, "")
    else:
        window = choose(ctx, dag, suspended, statement)
    ctx.loop.window = window
    return window


# -- choosing ----------------------------------------------------------------

def choose(ctx, dag: Any, suspended: Collection[str], query: str) -> Window:
    """The window ctx.settings.strategy chooses from dag, for query."""
    name, dial = ctx.settings.strategy, ctx.settings.strategy_dial
    lemmas = dag.lemmas
    eligible = [lid for lid in lemmas if lid not in suspended]
    size = ctx.settings.window_size
    n = len(eligible) if size < 0 else min(size, len(eligible))

    def newest(lid: str) -> Tuple[str, str]:
        return (str(lemmas[lid].get("proved_at") or ""), lid)

    if name == "recent":
        half = -1 if size < 0 else size // 2
        n_all, n_mine = (half if k is None else k
                         for k in (ctx.settings.lemmas_all, ctx.settings.my_lemmas))
        mine = [lid for lid in eligible if owner_of(lid) == ctx.user]
        return Window((tuple(_top(eligible, n_all, newest)),
                       tuple(_top(mine, n_mine, newest))))
    if name == "random":
        return Window((tuple(_sample(eligible, n, _exponent(dial), dag.depths(), newest)),))
    if name == "depth":
        return Window((tuple(_sample(_sinks(dag, eligible, suspended), n,
                                     _exponent(1.0 - dial), dag.depths(), newest)),))
    if name == "breadth":
        return Window((tuple(_breadth(dag, _sinks(dag, eligible, suspended), n, newest)),))
    if name == "relevance":
        return Window((tuple(_relevant(ctx, dag, eligible, n, query, newest)),), query)
    if name == "popular":
        cited_by = dag.cited_by_map()

        def citers(lid: str) -> Tuple[int, str, str]:
            return (sum(1 for c in cited_by.get(lid, ()) if c not in suspended),
                    *newest(lid))
        return Window((tuple(_top(eligible, n, citers)),))
    raise ValueError(f"unknown strategy {name!r}")


def _top(ids: List[str], n: int, key: Callable[[str], Any]) -> List[str]:
    """The n largest of ids by key, largest first; all of them for -1."""
    if n < 0 or n >= len(ids):
        return sorted(ids, key=key, reverse=True)
    return heapq.nlargest(n, ids, key=key)


def _exponent(dial: float) -> float:
    """The power of (depth + 1) a dial sets: 0 at 0, infinite at 1."""
    return math.inf if dial >= 1.0 else dial / (1.0 - dial)


def _sinks(dag: Any, eligible: List[str], suspended: Collection[str]) -> List[str]:
    """The eligible lemmas no unsuspended lemma cites."""
    cited_by = dag.cited_by_map()
    return [lid for lid in eligible
            if all(c in suspended for c in cited_by.get(lid, ()))]


def _sample(pool: List[str], n: int, exponent: float, depth: Mapping[str, int],
            newest: Callable[[str], Any]) -> List[str]:
    """n of pool, drawn without replacement with weight (depth + 1) **
    exponent: the n deepest, by recency among equals, for an infinite
    exponent. The weights depend on depth alone, so a draw picks a depth
    first (by its lemmas' total weight) and then one of its lemmas."""
    n = min(n, len(pool))
    if math.isinf(exponent):
        return _top(pool, n, lambda lid: (depth.get(lid, 0), *newest(lid)))
    levels: Dict[int, List[str]] = {}
    for lid in pool:
        levels.setdefault(depth.get(lid, 0), []).append(lid)
    top = max(levels, default=0)
    # Relative to the deepest level, in logs, so no power overflows.
    weight = {d: math.exp(exponent * (math.log(d + 1) - math.log(top + 1)))
              for d in levels}
    rng = random.Random()
    out: List[str] = []
    while len(out) < n:
        ds = list(levels)
        total = [len(levels[d]) * weight[d] for d in ds]
        d = rng.choices(ds, total)[0] if sum(total) > 0 else max(ds)
        bucket = levels[d]
        i = rng.randrange(len(bucket))
        bucket[i], bucket[-1] = bucket[-1], bucket[i]
        out.append(bucket.pop())
        if not bucket:
            del levels[d]
    return out


def _breadth(dag: Any, sinks: List[str], n: int,
             newest: Callable[[str], Any]) -> List[str]:
    """n sinks whose ancestries overlap least (see the module docstring),
    in the order chosen."""
    n = min(n, len(sinks))
    if n == 0:
        return []
    depth = dag.depths()
    start = max(sinks, key=lambda lid: (depth.get(lid, 0), *newest(lid)))
    np = Ancestry.numpy()
    pool = list(sinks)
    if np is None and len(pool) > BREADTH_POOL:
        pool = random.Random().sample(pool, BREADTH_POOL)
        if start not in pool:
            pool[0] = start
    sketches = dag.ancestry().sketches(dag.lemmas, pool)
    bare = [depth.get(lid, 0) == 0 for lid in pool]      # cites nothing
    bare_cap = n // 2
    first = pool.index(start)
    chosen = [first]
    bare_taken = int(bare[first])
    if np is not None:
        matrix = sketches

        def like(row: Any) -> Any:
            return np.count_nonzero(matrix == row, axis=1) / Ancestry.K

        worst = like(matrix[first])
        closed = np.zeros(len(pool), dtype=bool)
        closed[first] = True
        bare_mask = np.array(bare)
        while len(chosen) < n:
            if bare_taken >= bare_cap:
                closed |= bare_mask
            scores = np.where(closed, np.inf, worst)
            nxt = int(np.argmin(scores))
            if not np.isfinite(scores[nxt]):
                break
            chosen.append(nxt)
            closed[nxt] = True
            bare_taken += int(bare[nxt])
            worst = np.maximum(worst, like(matrix[nxt]))
    else:
        k = Ancestry.K

        def like(a: Sequence[int], b: Sequence[int]) -> float:
            return sum(x == y for x, y in zip(a, b)) / k

        worst = [like(s, sketches[first]) for s in sketches]
        closed = [False] * len(pool)
        closed[first] = True
        while len(chosen) < n:
            best, nxt = math.inf, -1
            for i, w in enumerate(worst):
                if closed[i] or (bare[i] and bare_taken >= bare_cap):
                    continue
                if w < best:
                    best, nxt = w, i
            if nxt < 0:
                break
            chosen.append(nxt)
            closed[nxt] = True
            bare_taken += int(bare[nxt])
            s = sketches[nxt]
            worst = [max(w, like(t, s)) for w, t in zip(worst, sketches)]
    return [pool[i] for i in chosen]


def _relevant(ctx, dag: Any, eligible: List[str], n: int, query: str,
              newest: Callable[[str], Any]) -> List[str]:
    """The n eligible lemmas that best match query (the search index's
    ranking); the newest instead, said once, when there is no index."""
    from ..tools.search_lemmas import run_index
    index = run_index(ctx)
    if index is None or index.wait():
        if "relevance" not in ctx.run.said:
            ctx.run.said.add("relevance")
            ctx.ui.log("note: --strategy relevance has no search index; "
                       "showing the newest lemmas instead.", err=True)
        return _top(eligible, n, newest)
    if not query.strip():
        return []
    wanted = set(eligible)
    return index.search(query, None if n >= len(eligible) else n,
                        wanted.__contains__)[:n]
