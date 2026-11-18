"""proofs export: a DAG, a user's lemmas, or one lemma and what it cites,
as a LaTeX document (latex.py), an interactive citation graph (html.py) or
a static image of it (png.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.context import Context
from ..core.dag import _citation_closure, _dependency_order, load_file, owner_of
from ..tools.run_code import computations
from ..workspace import layout as workspace
from . import graph, html, latex, png

FORMATS = ("latex", "html", "png")


def export_entry(
    *,
    lemma_id: Optional[str],
    owner: Optional[str],
    out_dir: str,
    root: str,
    user_id: str,
    parser: argparse.ArgumentParser,
    fmt: str,
    theme: str = "light",
) -> int:
    """proofs export NAME [LEMMA_ID] [--user USER_ID] --format FORMAT:
    write the chosen lemmas into out_dir (-o/--out, created if it is not
    there: the export is the user's artifact, not the project's data) as
    fmt — "latex", the document in dependency order (latex.py); "html",
    the interactive citation graph (html.py); "png", the static image of
    it in the given theme (png.py).

    With neither lemma_id nor owner it is the complete DAG; with owner, the
    lemmas that user's runs proved, ordered among themselves (their
    dependencies elsewhere are cited by id, not pulled in); with lemma_id,
    that lemma and everything it cites, the _citation_closure order — the
    order proofs verify --full verifies in. No model is involved: the
    document is a function of the files. Returns the exit code: 0 when the
    document was written.
    """
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
        name = latex.name_for(lemma_id=lemma_id)
        scope = f"lemma {lemma_id} and everything it cites — {len(order)} lemma(s)"
    elif owner is not None:
        order = _dependency_order(dag, {lid for lid in lemmas if owner_of(lid) == owner})
        name = latex.name_for(owner=owner)
        scope = f"{owner}'s lemmas — {len(order)} lemma(s)"
    else:
        order = _dependency_order(dag, set(lemmas))
        users = {owner_of(lid) for lid in order}
        name = latex.name_for()
        scope = (f"the complete DAG — {len(order)} lemma(s) from "
                 f"{len(users)} user(s)")
    if fmt != "latex":
        return _export_graph(ctx, dag, order, scope, paths.root.name,
                             Path(out_dir) / Path(name).with_suffix(
                                 ".png" if fmt == "png" else ".html").name,
                             fmt, theme, parser)
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
    out = latex.write(Path(out_dir) / name, document)
    ctx.ui.log(
        f"Exported {len(order)} lemma(s) to {out} — dependency order, "
        f"dependencies first."
    )
    return 0


def _export_graph(ctx: Context, dag, order, scope: str, conjecture_name: str,
                  out: Path, fmt: str, theme: str,
                  parser: argparse.ArgumentParser) -> int:
    """The citation graph of the lemmas in order, as the interactive page
    or the image. A suspended lemma is drawn hollow, so the picture says
    which results are not to be relied on yet."""
    lemmas = dag.lemmas
    suspended = {lid for lid in order if lid in dag.suspension()}
    laid_out = graph.layout(order, lemmas, suspended)
    if fmt == "png":
        try:
            png.render_to(out, laid_out, title=conjecture_name, theme=theme)
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
        )
        latex.write(out, page)
    ctx.ui.log(
        f"Exported the citation graph of {len(order)} lemma(s) "
        f"({len(laid_out['edges'])} citation(s)) to {out}."
    )
    return 0
