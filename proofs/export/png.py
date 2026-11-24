"""The static image of a DAG's citation graph: `proofs export NAME --format
png`, the picture export/html.py makes interactive, drawn once with
matplotlib.

matplotlib is an optional dependency (pip install 'proofs[png]'): only
this format needs it, so it is imported when an image is drawn, and its
absence is reported as a one-line error naming the extra (MissingDependency)
rather than a traceback.

    proof.png                      the complete DAG
    <user_id>_proof.png            a user's lemmas
    <user_id>--<run_id>--<hex>_proof.png  a lemma and everything it cites

The image is drawn at SCALE pixels a layout unit, and shrunk to fit when
that would make it larger than 3840x2160 (MAX_WIDTH, MAX_HEIGHT): a small
DAG gives a small image, a large one a monitor-size image. The theme is
light unless --theme asks for dark, proofs (the documentation site's
palette, its dark variant) or proofs-light (its light one): an image has no
viewer preference to follow, so the choice is made when it is drawn. A
lemma that settles the conjecture is ringed twice in the ink and always
labelled with what it settles.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Union

from . import graph as g

__all__ = ["MissingDependency", "render_to"]

# The image: SCALE pixels a layout unit, at most 3840x2160 (4K UHD) — the
# scale drops for a graph that would not fit. At least MIN_WIDTH wide, so
# the legend and the totals never meet over a small graph. Matplotlib
# sizes figures in inches, so at 100 dpi a pixel is 0.72 pt.
SCALE = 2.0
MAX_WIDTH = 3840
MAX_HEIGHT = 2160
MIN_WIDTH = 1440
DPI = 100
PT = 72 / DPI
# Margins in pixels: the legend and totals above (TOP at least, more for a
# long legend), room for stacked labels below, and the sides.
TOP = 200.0
SIDE = 80.0
BOTTOM = 160.0
# Text and strokes, in pixels.
LABEL = 20.0
LEGEND = 26.0
EDGE_PX = 1.2
# A settling lemma's two rings: their offsets past its radius, and width.
SETTLE_RINGS = (6.0, 12.0)
SETTLE_PX = 2.6


def _lemmas(count: int) -> str:
    return f"{count} lemma" + ("" if count == 1 else "s")


class MissingDependency(RuntimeError):
    """matplotlib is not installed."""


def _bezier(x1: float, y1: float, x2: float, y2: float):
    """An edge's cubic: leaving the citer straight down, arriving at the
    cited lemma straight down, the way html.py draws it."""
    dy = (y2 - y1) / 2
    return [(x1, y1), (x1, y1 + dy), (x2, y2 - dy), (x2, y2)]


def render_to(path: Union[str, os.PathLike], graph: Mapping[str, Any], *,
              title: str, theme: str = "light") -> Path:
    """Draw graph (graph.layout's result) to a PNG at path, written the way
    latex.write writes: a temp file beside it, then os.replace."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import EllipseCollection, PathCollection
        from matplotlib.path import Path as MPath
    except ImportError as e:
        raise MissingDependency(
            "--format png needs matplotlib: pip install 'proofs[png]' "
            "(or matplotlib on its own)") from e

    t = g.THEMES[theme]
    nodes = graph["nodes"]
    edges = graph["edges"]
    # The legend: a dot and "user (n lemmas)" per colour, then the hollow
    # ring for a suspended lemma and the double ring for a settling one,
    # when there is one. Each entry is (colour, text, mark), mark "fill",
    # "hollow" or "settle".
    users = graph["users"]
    shown = users[:len(g.CATEGORICAL[theme])]
    entries = [(g.colour(theme, k), f"{u['id']} ({_lemmas(u['count'])})", "fill")
               for k, u in enumerate(shown)]
    if len(users) > len(shown):
        rest = sum(u["count"] for u in users[len(shown):])
        entries.append((g.OTHER[theme],
                        f"{len(users) - len(shown)} other users ({_lemmas(rest)})", "fill"))
    if any(nd["suspended"] for nd in nodes):
        entries.append((t["ink2"], "suspended", "hollow"))
    for kind in ("proved", "disproved"):
        if any(nd.get("settles") == kind for nd in nodes):
            entries.append((t["ink2"], g.SETTLES_TEXT[kind], "settle"))
    step = LEGEND * 1.6
    top = max(TOP, step * len(entries) + 100)

    # Layout units to pixels: SCALE, or less when that would pass the
    # maximum size; the image is then sized to the graph, centred in it.
    gw, gh = graph["width"], graph["height"]
    s = min(SCALE, (MAX_WIDTH - 2 * SIDE) / max(gw, 1.0),
            (MAX_HEIGHT - top - BOTTOM) / max(gh, 1.0))
    width = round(max(gw * s + 2 * SIDE, MIN_WIDTH))
    height = round(gh * s + top + BOTTOM)
    ox = (width - gw * s) / 2
    oy = top

    def X(x: float) -> float:
        return ox + x * s

    def Y(y: float) -> float:
        return oy + y * s

    # A node's radius scales with the graph (the layout keeps neighbours
    # apart at any scale), never under 1.5 px.
    def radius(r: float) -> float:
        return max(1.5, r * s)

    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    fig.patch.set_facecolor(t["surface"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor(t["surface"])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    if edges:
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
        paths = [MPath(_bezier(X(nodes[a]["x"]), Y(nodes[a]["y"]),
                               X(nodes[b]["x"]), Y(nodes[b]["y"])), codes)
                 for a, b in edges]
        ax.add_collection(PathCollection(
            paths, facecolors="none", edgecolors=t["edge"],
            alpha=t["edge_alpha"] * 1.6, linewidths=EDGE_PX * PT, zorder=1))

    rad = [radius(nd["r"]) for nd in nodes]
    if nodes:
        sizes = [2 * r for r in rad]
        ax.add_collection(EllipseCollection(
            sizes, sizes, [0] * len(nodes), units="xy",
            offsets=[(X(nd["x"]), Y(nd["y"])) for nd in nodes],
            offset_transform=ax.transData,
            facecolors=[t["surface"] if nd["suspended"] else g.colour(theme, nd["owner"])
                        for nd in nodes],
            edgecolors=[g.colour(theme, nd["owner"]) if nd["suspended"] else t["surface"]
                        for nd in nodes],
            linewidths=[1.6 * PT if nd["suspended"] else 1.0 * PT for nd in nodes],
            zorder=2))
        settling = [i for i, nd in enumerate(nodes) if nd.get("settles")]
        if settling:
            rings = [(i, off) for i in settling for off in SETTLE_RINGS]
            diam = [2 * (rad[i] + off) for i, off in rings]
            ax.add_collection(EllipseCollection(
                diam, diam, [0] * len(rings), units="xy",
                offsets=[(X(nodes[i]["x"]), Y(nodes[i]["y"])) for i, _ in rings],
                offset_transform=ax.transData, facecolors="none",
                edgecolors=t["settle"], linewidths=SETTLE_PX * PT, zorder=2.5))

    # The permanent labels, placed greedily the way html.py places them:
    # the first of three steps away from the node that overlaps no label
    # already placed. Widths are estimated (monospace, 0.62 em a character).
    boxes = []
    h = LABEL * 1.5
    for i in g.labelled(graph):
        nd = nodes[i]
        x, y0 = X(nd["x"]), Y(nd["y"])
        settles = nd.get("settles")
        # A settling lemma's label goes above its rings, in the ink.
        top_row = bool(settles) or (nd["row"] == graph["rows"] - 1
                                    and nd["cited_by"] == 0)
        if settles:
            text = f"{g.short_id(nd['id'])} · {g.SETTLES_TEXT[settles]}"
        else:
            text = g.short_id(nd["id"]) + ("" if top_row else f" ({nd['cited_by']})")
        w = 0.62 * LABEL * len(text) + 10
        reach = rad[i] + (SETTLE_RINGS[-1] if settles else 0.0)
        for k in range(3):
            off = reach + 12 + k * h
            y = y0 - off if top_row else y0 + off
            box = (x - w / 2, y - h if top_row else y, w, h)
            if any(b[0] < box[0] + box[2] and box[0] < b[0] + b[2]
                   and b[1] < box[1] + box[3] and box[1] < b[1] + b[3]
                   for b in boxes):
                continue
            boxes.append(box)
            ax.text(x, y, text, ha="center", va="bottom" if top_row else "top",
                    fontsize=LABEL * PT, zorder=3, family="monospace",
                    color=t["ink"] if settles else t["ink2"],
                    bbox={"facecolor": t["surface"], "edgecolor": "none",
                          "alpha": 0.85, "pad": 2})
            break

    strip = [i for i, nd in enumerate(nodes) if nd["row"] < 0]
    if strip:
        first = min(strip, key=lambda i: nodes[i]["x"])
        x, y = X(nodes[first]["x"]), Y(gh)
        text = "not yet connected"
        style = {"fontsize": LABEL * PT, "style": "italic", "color": t["muted"]}
        if x - 20 - 0.55 * LABEL * len(text) >= SIDE / 2:
            ax.text(x - 20, y, text, ha="right", va="center", **style)
        else:
            # No room to its left (the strip is the widest row): above it.
            ax.text(x - rad[first], y - rad[first] - 10, text, ha="left",
                    va="bottom", **style)

    y = 50.0
    for col, text, mark in entries:
        if mark == "settle":
            # A small filled dot inside the two rings, as on the graph.
            ax.scatter([50], [y], s=(LEGEND * 0.3 * PT) ** 2, zorder=3,
                       facecolors=col, edgecolors="none")
            for d in (LEGEND * 0.55, LEGEND * 0.85):
                ax.scatter([50], [y], s=(d * PT) ** 2, zorder=3,
                           facecolors="none", edgecolors=t["settle"],
                           linewidths=1.6 * PT)
        else:
            ax.scatter([50], [y], s=(LEGEND * 0.55 * PT) ** 2, zorder=3,
                       facecolors=t["surface"] if mark == "hollow" else col,
                       edgecolors=col, linewidths=2 * PT)
        ax.text(72, y, text, va="center", fontsize=LEGEND * PT,
                family="monospace", color=t["ink"])
        y += step
    ax.text(width - 50, 50 - LEGEND * 0.6,
            f"{title}\n{_lemmas(len(nodes))} · {len(edges)} "
            f"citation{'' if len(edges) == 1 else 's'}\n"
            f"node size ∝ times cited",
            ha="right", va="top", fontsize=LEGEND * PT, color=t["ink2"],
            linespacing=1.5)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    fig.savefig(tmp, format="png", dpi=DPI, facecolor=t["surface"])
    plt.close(fig)
    os.replace(tmp, p)
    return p
