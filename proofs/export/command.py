"""proofs export: a DAG, a user's lemmas, or one lemma and what it cites,
as a LaTeX document (latex.py), an interactive citation graph (html.py) or
a static image of it (png.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from ..core import resolution
from ..core.context import Context
from ..core.dag import _citation_closure, _dependency_order, load_file, owner_of
from ..tools.run_code import computations
from ..workspace import layout as workspace
from . import graph, html, latex, png

FORMATS = ("latex", "html", "png")
# The file extension each format writes.
EXTENSIONS = {"latex": ".tex", "html": ".html", "png": ".png"}
# --theme, by format. The page's "proofs" is the docs' palette following
# the viewer's OS setting, as the docs site does; an image has no viewer
# to follow, so it names the variant: "proofs" (dark, the site's default)
# or "proofs-light".
THEMES = {"html": ("light", "dark", "proofs"),
          "png": ("light", "dark", "proofs", "proofs-light")}


def file_name(default: str, name: Optional[str], fmt: str) -> str:
    """The export's file name in the -o/--out directory: -n/--name when
    given, else the default (latex.name_for's, its stem kept), with fmt's
    extension. A name already ending in that extension keeps it; any other
    suffix is part of the name ("v1.2" becomes "v1.2.tex"). Raises
    ValueError for a name that is a path, or that ends in another format's
    extension (writing an image as "graph.html" is a mistake, not a choice).
    """
    ext = EXTENSIONS[fmt]
    if name is None:
        return Path(default).stem + ext
    if not name.strip() or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(
            f"-n/--name takes a file name, not a path: {name!r}; "
            f"-o/--out names the directory"
        )
    suffix = Path(name).suffix.lower()
    if suffix == ext:
        return name
    if suffix in EXTENSIONS.values() or suffix == ".htm":
        raise ValueError(
            f"-n/--name {name!r} ends in {suffix}, but --format {fmt} "
            f"writes {ext}"
        )
    return name + ext



def export_entry(
    *,
    lemma_id: Optional[str],
    owner: Optional[str],
    out_dir: str,
    root: str,
    user_id: str,
    parser: argparse.ArgumentParser,
    fmt: str,
    theme: Optional[str] = None,
    name: Optional[str] = None,
) -> int:
    """proofs export NAME [LEMMA_ID] [--user USER_ID] --format FORMAT:
    write the chosen lemmas into out_dir (-o/--out, created if it is not
    there: the export is the user's artifact, not the project's data) as
    fmt — "latex", the document in dependency order (latex.py); "html",
    the interactive citation graph (html.py), opening in the given theme
    (light, dark or proofs; default proofs, which follows the viewer's OS
    setting); "png", the static image of it in the given theme (light,
    dark, proofs or proofs-light; default light) (png.py). The file is named name (-n/--name,
    see file_name) or, without it, after what is exported.

    With neither lemma_id nor owner it is the complete DAG; with owner, the
    lemmas that user's runs proved, ordered among themselves (their
    dependencies elsewhere are cited by id, not pulled in); with lemma_id,
    that lemma and everything it cites, the _citation_closure order — the
    order proofs verify --full verifies in. No model is involved: the
    document is a function of the files. Returns the exit code: 0 when the
    document was written.
    """
    try:
        file_name("proof.tex", name, fmt)
    except ValueError as e:
        parser.error(str(e))
    ctx = Context(user=user_id)
    try:
        paths = workspace.resolve(root, user_id)
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
    ctx.paths.conjecture_root = str(paths.root)
    ctx.paths.dags_dir = str(paths.dags_dir)
    ctx.paths.references_file = str(paths.references)
    ctx.ui.log(workspace.describe(paths))
    dag = ctx.dag
    lemmas = dag.lemmas
    if lemma_id is not None:
        if lemma_id not in lemmas:
            parser.error(f"{lemma_id} is not a lemma of the DAG {root}")
        order = _citation_closure(dag, lemma_id)
        default = latex.name_for(lemma_id=lemma_id)
        scope = f"lemma {lemma_id} and everything it cites — {len(order)} lemma(s)"
    elif owner is not None:
        order = _dependency_order(dag, {lid for lid in lemmas if owner_of(lid) == owner})
        default = latex.name_for(owner=owner)
        scope = f"{owner}'s lemmas — {len(order)} lemma(s)"
    else:
        order = _dependency_order(dag, set(lemmas))
        users = {owner_of(lid) for lid in order}
        default = latex.name_for()
        scope = (f"the complete DAG — {len(order)} lemma(s) from "
                 f"{len(users)} user(s)")
    out_path = Path(out_dir) / file_name(default, name, fmt)
    if fmt != "latex":
        return _export_graph(ctx, dag, order, scope, paths.root.name,
                             load_file(paths.conjecture), out_path, fmt,
                             theme, parser)
    # The scripts the exported lemmas rest on, for the appendix: latex.py
    # renders from data, so they are read here.
    run_lines = computations.load_runs(paths.root)
    code_shown: Dict[str, Dict[str, Any]] = {}
    for lid in order:
        for e in computations.entries(lemmas.get(lid) or {}):
            code_shown[e["hash"]] = {
                "lang": e["lang"],
                "produced_output": e.get("produced_output"),
                "script": computations.load_script(paths.root, e),
                "ran_with": sorted({
                    f"{(l.get('tool') or {}).get('name')} "
                    f"{(l.get('tool') or {}).get('version')}"
                    for l in computations.executions(paths.root, e["hash"], run_lines)
                }),
            }
    # The in-memory DAG keeps no proofs: read the exported ones back.
    proofs = dag.proofs(list(order))
    document = latex.render(
        conjecture_name=paths.root.name,
        conjecture=load_file(paths.conjecture),
        scope=scope,
        order=order,
        lemmas={lid: {**lemmas[lid], "proof": proofs[lid]} for lid in order},
        references=dag.references,
        computations=code_shown,
    )
    out = latex.write(out_path, document)
    ctx.ui.log(
        f"Exported {len(order)} lemma(s) to {out} — dependency order, "
        f"dependencies first."
    )
    return 0


def _settles(ctx: Context, dag, order, conjecture: str) -> Dict[str, str]:
    """The exported lemmas that settle the conjecture, with the kind,
    "proved" or "disproved": resolution.find over the complete DAG, the
    test a run makes, so the picture marks exactly what proofs status
    reports. A DAG holding both kinds marks both, and says so."""
    found = resolution.find(dag.lemmas, dag.suspension().suspended, conjecture)
    if len({r.kind for r in found}) > 1:
        ctx.ui.log(
            "The DAG both proves and disproves the conjecture ("
            + "; ".join(r.lemma_id for r in found)
            + "): at least one of these proofs is wrong. Both are marked.")
    exported = set(order)
    return {r.lemma_id: r.kind for r in found if r.lemma_id in exported}


def _export_graph(ctx: Context, dag, order, scope: str, conjecture_name: str,
                  conjecture: str, out: Path, fmt: str, theme: Optional[str],
                  parser: argparse.ArgumentParser) -> int:
    """The citation graph of the lemmas in order, as the interactive page
    or the image. A suspended lemma is drawn hollow, so the picture says
    which results are not to be relied on yet; a lemma that settles the
    conjecture is ringed and always labelled, so it says what is done."""
    lemmas = dag.lemmas
    suspended = {lid for lid in order if lid in dag.suspension()}
    settles = _settles(ctx, dag, order, conjecture)
    laid_out = graph.layout(order, lemmas, suspended, settles)
    if fmt == "png":
        try:
            png.render_to(out, laid_out, title=conjecture_name,
                          theme=theme or "light")
        except png.MissingDependency as e:
            parser.error(str(e))
    else:
        proofs = dag.proofs(list(order))
        page = html.render(
            graph=laid_out,
            labels=graph.labelled(laid_out),
            lemmas={lid: {**lemmas[lid], "proof": proofs[lid]} for lid in order},
            conjecture_name=conjecture_name,
            scope=scope,
            theme=theme or "proofs",
        )
        latex.write(out, page)
    for lid, kind in settles.items():
        ctx.ui.log(f"Marked {lid}: it {graph.SETTLES_TEXT[kind]}.")
    ctx.ui.log(
        f"Exported the citation graph of {len(order)} lemma(s) "
        f"({len(laid_out['edges'])} citation(s)) to {out}."
    )
    return 0
