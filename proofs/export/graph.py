"""The citation graph of a DAG, laid out for drawing: what export/html.py
and export/png.py both draw, so the interactive page and the static image
show the same picture.

The layout is layered, dependencies at the bottom:

    row     a lemma's row is the longest citation chain beneath it among
            the exported lemmas: 0 for one citing none of them. Every
            citation therefore runs from a higher row down to a lower one,
            so whether an edge is "cites" or "cited by" reads off the
            picture's vertical direction.
    strip   a lemma no exported lemma cites and that cites none of them is
            not yet connected: it goes in a strip under the rows rather
            than padding row 0.
    order   within a row, lemmas are ordered by a few barycenter sweeps —
            each lemma moved toward the mean position of its neighbours,
            alternately those below and those above — which untangles
            most crossings at O(edges) a sweep.
    size    a node's area is proportional to how many exported lemmas cite
            it.
    colour  the user whose run proved it (owner_of), in a fixed order:
            users by lemma count, most first, so a user keeps its colour
            between exports of the same DAG unless the counts reorder.
    settles a lemma that settles the conjecture (resolution.find) is
            marked "proved" or "disproved", for the renderers to ring and
            always label.

Four themes, each a categorical palette and chart chrome: "light" and
"dark", neutral; "proofs" and "proofs-light", the documentation site's
palette (docs/assets/style.css: midnight slate, cerulean and rose — dark
by default, light under a light OS setting), so a graph embedded in the
site or shown beside it reads as part of it.

Coordinates are abstract units — x in [0, width], y in [0, height], y
growing downward — that each renderer scales to its own surface. Nothing
here depends on a drawing library; the layout is a function of the
exported lemmas, so two exports of the same files are the same picture.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from ..core.dag import owner_of

__all__ = ["CATEGORICAL", "SETTLES_TEXT", "THEMES", "colour", "labelled", "layout",
           "short_id"]

# Abstract units: the horizontal distance between neighbours' centres in a
# row (at least: big neighbours are pushed further apart, GAP between their
# edges), and the vertical distance between rows.
COL = 24.0
GAP = 8.0
ROW = 64.0
# Node radii: R0 for a lemma nothing cites, growing with the square root of
# the times cited (area proportional to it), capped at R_MAX.
R0 = 3.0
R_PER = 2.2
R_MAX = 30.0
SWEEPS = 8

# The categorical slots, in this order. Users past the eighth share the
# "other" colour rather than a generated hue. "light" and "dark" are the
# dataviz reference palette; the two "proofs" palettes start from the
# docs' cerulean (#2C639E), rose (#B8305F / #F04D86) and green, with the
# docs' cream for the third slot, and were stepped until they pass the
# adjacent-pair CVD separation check (>= 8) against the docs' surfaces,
# #121925 and #FFFFFF. The cream is a deliberate exception to the
# lightness band and chroma floor (it is pale and near-neutral by
# design), and on white (proofs-light) it is a deeper sand at 2.2:1, the
# palest that still reads as a node. The page's stylesheet
# (page/graph.css) carries the same values: change both together.
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"],
    "proofs": ["#3B7AC0", "#E0457D", "#E6D8B8", "#1FA67E",
               "#8C6FE0", "#2FA3C4", "#E5683B", "#B85CC9"],
    "proofs-light": ["#2C639E", "#C2386A", "#C4AE80", "#16936A",
                     "#6B4FD0", "#1F86A6", "#D2582A", "#A04AB5"],
}
OTHER = {"light": "#898781", "dark": "#898781",
         "proofs": "#6F7D92", "proofs-light": "#8A93A1"}
# The chart chrome both renderers draw with. settle is the double ring
# around a lemma that settles the conjecture: the ink, never a hue, since
# a hue on a node means its user.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "edge": "#0b0b0b", "edge_alpha": 0.10,
              "highlight": "#0b0b0b", "settle": "#0b0b0b"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "muted": "#898781", "edge": "#ffffff", "edge_alpha": 0.12,
             "highlight": "#ffffff", "settle": "#ffffff"},
    "proofs": {"surface": "#121925", "ink": "#F3EEE4", "ink2": "#BDBDBD",
               "muted": "#92A2B8", "edge": "#DCEBFA", "edge_alpha": 0.12,
               "highlight": "#F3EEE4", "settle": "#F3EEE4"},
    "proofs-light": {"surface": "#FFFFFF", "ink": "#18202C", "ink2": "#4F5866",
                     "muted": "#646D7C", "edge": "#1B2B45", "edge_alpha": 0.10,
                     "highlight": "#18202C", "settle": "#18202C"},
}
# What a settling lemma's label and legend entry say, by resolution kind.
SETTLES_TEXT = {"proved": "proves the conjecture",
                "disproved": "disproves the conjecture"}


def short_id(lemma_id: str) -> str:
    """A lemma id as a label: <user>/<run>/<hex> becomes <user>/…/<hex>,
    since the run id is long and rarely what a reader is looking for."""
    parts = str(lemma_id).split("/")
    if len(parts) >= 3:
        return f"{parts[0]}/…/{parts[-1]}"
    return str(lemma_id)


def _cited(node: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for entry in node.get("cited_lemmas") or []:
        if isinstance(entry, str) and entry.strip() and entry not in out:
            out.append(entry)
    return out


def _step(r: List[float], a: int, b: int) -> float:
    """The distance between neighbours a and b's centres."""
    return max(COL, r[a] + r[b] + GAP)


def _extent(row: List[int], r: List[float]) -> float:
    return sum(_step(r, a, b) for a, b in zip(row, row[1:]))


def _spread(row: List[int], x: List[float], r: List[float], width: float) -> None:
    """Place a row's nodes, in their current order, _step apart and
    centred on the picture's width."""
    pos = (width - _extent(row, r)) / 2
    for j, node in enumerate(row):
        if j:
            pos += _step(r, row[j - 1], node)
        x[node] = pos


def layout(
    order: Sequence[str],
    lemmas: Mapping[str, Mapping[str, Any]],
    suspended: Optional[Set[str]] = None,
    settles: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """The laid-out graph of the lemmas in order (dependency order, as
    export_entry settles it). A citation of a lemma outside order is not
    an edge (it is kept in the node's "external" list); one that would
    run upward — only a citation cycle, which _post_order cuts — is kept
    as an edge but not used for the rows. settles maps the lemmas that
    settle the conjecture to their kind, "proved" or "disproved".

    Returns {"nodes", "edges", "users", "rows", "width", "height"}:
    nodes in order, each {"id", "owner" (index into users), "row" (-1 for
    the strip), "x", "y", "r", "cited_by" (count), "suspended", "settles"
    (the kind, or None)}; edges as
    [citer index, cited index]; users as [{"id", "count"}], the colour
    order.
    """
    ids = [lid for lid in order if isinstance(lemmas.get(lid), Mapping)]
    index = {lid: i for i, lid in enumerate(ids)}
    n = len(ids)
    cites: List[List[int]] = [[] for _ in range(n)]
    external: List[List[str]] = [[] for _ in range(n)]
    for i, lid in enumerate(ids):
        for c in _cited(lemmas[lid]):
            if c in index:
                cites[i].append(index[c])
            else:
                external[i].append(c)
    citers: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for c in cites[i]:
            citers[c].append(i)

    # Rows, in one pass: order puts every lemma after what it cites, except
    # across a cut cycle, whose back edge is skipped here.
    row = [0] * n
    for i in range(n):
        row[i] = 1 + max((row[c] for c in cites[i] if c < i), default=-1)
    connected = [bool(cites[i] or citers[i]) for i in range(n)]
    for i in range(n):
        if not connected[i]:
            row[i] = -1
    n_rows = 1 + max(row, default=-1)
    rows: List[List[int]] = [[] for _ in range(n_rows)]
    strip: List[int] = []
    for i in range(n):
        (rows[row[i]] if row[i] >= 0 else strip).append(i)

    rad = [min(R0 + R_PER * len(citers[i]) ** 0.5, R_MAX) for i in range(n)]
    width = COL + max([_extent(r, rad) for r in rows] + [_extent(strip, rad)])
    x = [0.0] * n
    for r in rows:
        _spread(r, x, rad, width)
    # Barycenter sweeps: upward, each row ordered by the mean x of what its
    # lemmas cite (all below it); then downward, by the mean x of what
    # cites them (all above). A lemma with no neighbours on that side keeps
    # its place. Stable sorts on (key, current position) keep it
    # deterministic.
    for sweep in range(SWEEPS):
        up = sweep % 2 == 0
        seq = range(1, n_rows) if up else range(n_rows - 2, -1, -1)
        for k in seq:
            r = rows[k]
            key: Dict[int, float] = {}
            for node in r:
                nbrs = cites[node] if up else citers[node]
                key[node] = (sum(x[m] for m in nbrs) / len(nbrs)) if nbrs else x[node]
            r.sort(key=lambda m: (key[m], x[m]))
            _spread(r, x, rad, width)
    # The strip: by owner, so its colours read as runs, then by id.
    owners = [owner_of(lid) for lid in ids]
    strip.sort(key=lambda m: (owners[m], ids[m]))
    _spread(strip, x, rad, width)

    counts = Counter(owners)
    users = sorted(counts, key=lambda u: (-counts[u], u))
    user_index = {u: k for k, u in enumerate(users)}
    # y grows downward: the top row first, row 0 at the bottom, the strip
    # one and a half rows under it.
    height = ROW * max(n_rows - 1, 0) + (1.5 * ROW if strip else 0.0)
    suspended = suspended or set()
    settles = settles or {}
    nodes = []
    for i, lid in enumerate(ids):
        y = ROW * (n_rows - 1 - row[i]) if row[i] >= 0 else height
        k = len(citers[i])
        nodes.append({
            "id": lid,
            "owner": user_index[owners[i]],
            "row": row[i],
            "x": round(x[i], 2),
            "y": round(y, 2),
            "r": round(rad[i], 2),
            "cited_by": k,
            "suspended": lid in suspended,
            "settles": settles.get(lid),
            "external": external[i],
        })
    edges = [[i, c] for i in range(n) for c in cites[i]]
    return {
        "nodes": nodes,
        "edges": edges,
        "users": [{"id": u, "count": counts[u]} for u in users],
        "rows": n_rows,
        "width": width,
        "height": height,
    }


def labelled(graph: Mapping[str, Any], most_cited: int = 3, top: int = 3) -> List[int]:
    """The nodes worth a permanent label: first any lemma that settles the
    conjecture, always; then the most_cited most-cited (when anything
    cites them), and the top row's lemmas when it has at most top of them
    — the results everything else builds toward. The renderers place
    labels greedily in this order, so a settling lemma's is never crowded
    out."""
    nodes = graph["nodes"]
    out = [i for i, nd in enumerate(nodes) if nd.get("settles")]
    ranked = sorted((i for i, nd in enumerate(nodes) if nd["cited_by"]),
                    key=lambda i: (-nodes[i]["cited_by"], nodes[i]["id"]))
    out += [i for i in ranked[:most_cited] if i not in out]
    top_row = [i for i, nd in enumerate(nodes)
               if nd["row"] == graph["rows"] - 1 and graph["rows"] > 1]
    if len(top_row) <= top:
        out += [i for i in top_row if i not in out]
    return out


def colour(theme: str, owner: int) -> str:
    """A user's colour in a theme: its categorical slot, or "other" past
    the eighth."""
    slots = CATEGORICAL[theme]
    return slots[owner] if owner < len(slots) else OTHER[theme]
