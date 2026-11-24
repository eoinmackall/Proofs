"""The interactive citation graph of a DAG: `proofs export NAME --format
html`, one self-contained .html file.

The page draws graph.py's layout on a canvas: hovering a lemma highlights
every edge into and out of it (one colour — what it cites is always below
it, what cites it above) and shows its statement; clicking pins it in a
side panel with its proof, what it cites and what cites it, and an option
to highlight its full chain. Wheel or pinch zooms, dragging pans, the
search box finds a lemma by id or hex suffix. A lemma that settles the
conjecture is ringed twice and always labelled; a suspended lemma
(exported only with --suspended) is red and ringed in red, and its panel
says why it is suspended.

Three themes, on a Light / Dark / Theme switch: neutral light, neutral
dark, and Theme — the documentation site's palette (docs/assets/style.css),
dark or light by the viewer's OS setting the way the site itself is. The
page opens in the one --theme names (default Theme) and remembers the
viewer's pick, per browser, when the browser allows it.

Everything the page needs is inside it: the layout and the lemmas' text as
JSON, the page's own script and stylesheet (export/page/), and KaTeX
(export/vendor/katex) with its fonts as data URIs — so the file opens
offline, from anywhere, and is a function of the files like the LaTeX
export. KaTeX's script and stylesheet are MIT licensed and its fonts SIL
OFL 1.1; both licences ask that their notice travel with every copy, so
the page carries both texts in a comment at its top:

    proof.html                      the complete DAG
    <user_id>_proof.html            a user's lemmas
    <user_id>--<run_id>--<hex>_proof.html  a lemma and everything it cites
"""

from __future__ import annotations

import base64
import html
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["THEMES", "render"]

# The switch's themes, as data-theme values, with their button labels.
THEMES = {"light": "Light", "dark": "Dark", "proofs": "Theme"}

_HERE = Path(__file__).parent
_PAGE = _HERE / "page"
_KATEX = _HERE / "vendor" / "katex"

# A KaTeX @font-face src list: woff2, woff and ttf in turn. Only the woff2
# is kept — every browser that runs the page's script reads it — inlined
# as a data URI.
_FONT_SRC = re.compile(
    r'src:url\(fonts/([A-Za-z0-9_-]+)\.woff2\) format\("woff2"\)'
    r'(?:,url\([^)]*\) format\("[^"]*"\))*'
)


@lru_cache(maxsize=1)
def _notices() -> str:
    """The third-party notices the page must carry, as one HTML comment:
    KaTeX's MIT licence and its fonts' OFL, verbatim."""
    text = "\n\n".join([
        "This page embeds KaTeX 0.19.1 (https://katex.org), unmodified.",
        "katex.min.js and katex.min.css:\n\n"
        + (_KATEX / "LICENSE").read_text(encoding="utf-8").strip(),
        "The KaTeX fonts (embedded as data URIs):\n\n"
        + (_KATEX / "fonts" / "OFL.txt").read_text(encoding="utf-8").strip(),
    ])
    # A comment ends at the first "-->": none is in either licence, but a
    # notice must never be able to end it early.
    return "<!--\n" + text.replace("-->", "- ->") + "\n-->"


@lru_cache(maxsize=1)
def _katex() -> Dict[str, str]:
    """KaTeX's script and its stylesheet, fonts inlined."""
    def font(m: "re.Match[str]") -> str:
        data = (_KATEX / "fonts" / f"{m.group(1)}.woff2").read_bytes()
        return ('src:url(data:font/woff2;base64,'
                + base64.b64encode(data).decode("ascii") + ') format("woff2")')
    css = _FONT_SRC.sub(font, (_KATEX / "katex.min.css").read_text(encoding="utf-8"))
    return {"js": (_KATEX / "katex.min.js").read_text(encoding="utf-8"), "css": css}


def _script_json(value: Any) -> str:
    """JSON safe inside a <script> element: a "</" in a lemma's text would
    otherwise end the element early, and U+2028/2029 are line ends to
    older parsers."""
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            .replace("</", "<\\/").replace(" ", "\\u2028")
            .replace(" ", "\\u2029"))


def _cited_refs(node: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for r in node.get("cited_references") or []:
        if isinstance(r, str) and r.strip() and r not in out:
            out.append(r)
    return out


def render(
    *,
    graph: Mapping[str, Any],
    labels: List[int],
    lemmas: Mapping[str, Mapping[str, Any]],
    conjecture_name: str,
    scope: str,
    theme: str = "proofs",
    suspended: Optional[Mapping[str, Sequence[str]]] = None,
) -> str:
    """The page for graph (graph.layout's result) with labels (graph.labelled's
    nodes), lemmas mapping each id to its node with its proof, opening in
    theme (a THEMES key). suspended maps each suspended lemma to why it is
    suspended, for its panel."""
    suspended = suspended or {}
    if theme not in THEMES:
        raise ValueError(f"the page's theme is one of {', '.join(THEMES)}, not {theme!r}")
    nodes = []
    for nd in graph["nodes"]:
        node = lemmas.get(nd["id"]) or {}
        nodes.append({
            "id": nd["id"], "o": nd["owner"], "row": nd["row"],
            "x": nd["x"], "y": nd["y"], "r": nd["r"], "n": nd["cited_by"],
            "s": nd["suspended"], "res": nd.get("settles"), "ext": nd["external"],
            "why": list(suspended.get(nd["id"], ())) if nd["suspended"] else [],
            "refs": _cited_refs(node),
            "st": str(node.get("statement") or "").strip(),
            "pf": str(node.get("proof") or "").strip(),
        })
    data = {
        "width": graph["width"], "height": graph["height"], "rows": graph["rows"],
        "users": graph["users"], "nodes": nodes, "edges": graph["edges"],
        "labels": labels, "strip": any(nd["row"] < 0 for nd in graph["nodes"]),
    }
    switch = "".join(
        f'<button type="button" data-theme="{key}" '
        f'aria-pressed="{"true" if key == theme else "false"}">{label}</button>'
        for key, label in THEMES.items())
    k = _katex()
    title = html.escape(conjecture_name)
    return "\n".join([
        "<!doctype html>",
        _notices(),
        f'<html lang="en" data-theme="{theme}">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{title} · citation graph</title>",
        f"<style>{k['css']}</style>",
        f"<style>{(_PAGE / 'graph.css').read_text(encoding='utf-8')}</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<h1>{title}</h1>",
        f'<span class="scope">{html.escape(scope)}</span>',
        '<span class="spacer"></span>',
        '<input id="search" class="mono" list="ids" placeholder="Find a lemma by id…" '
        'aria-label="Find a lemma" autocomplete="off">',
        '<datalist id="ids"></datalist>',
        '<button id="fit" title="Fit the graph to the window (0)">Fit</button>',
        f'<div id="theme" class="seg" role="group" aria-label="Colour theme">{switch}</div>',
        "</header>",
        "<main>",
        '<div id="stage"><canvas id="base"></canvas><canvas id="over"></canvas>'
        '<div id="legend"></div><div id="tip" hidden></div></div>',
        '<aside id="panel" hidden aria-label="Lemma"></aside>',
        "</main>",
        f'<script type="application/json" id="data">{_script_json(data)}</script>',
        f"<script>{k['js']}</script>",
        f"<script>{(_PAGE / 'graph.js').read_text(encoding='utf-8')}</script>",
        "</body>",
        "</html>",
        "",
    ])
