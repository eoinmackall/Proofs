"""The LaTeX export of a DAG.

`proofs export NAME [LEMMA_ID] [--user USER_ID] --format latex` writes a LaTeX document of
the DAG, in dependency order: every lemma comes after the lemmas it cites,
so the document reads as a sequence of results each resting on the ones
above it. What it is given is what the document holds:

    NAME                -> the complete DAG
    NAME --user USER_ID -> the lemmas that user's runs proved
    NAME LEMMA_ID       -> that lemma and everything it cites

The document's body is the lemmas' own text: each statement in a lemma
environment, each proof in a proof environment, written verbatim. The
model writes statements and proofs as LaTeX text and the Merkle hash
covers them as-is, so nothing here rewrites or escapes them — a lemma id
in a heading is escaped, since an id is an identifier, not math. A lemma's
citations are listed under it: a cited lemma that is in the document is
cross-referenced to its section, one the document does not hold (another
user's lemma in a single-user export, or a dangling citation) is shown by
its id, and the references the document's lemmas cite are collected in a
References section at the end, each with its slogan, formal statement and
source.

The output file is written into the directory the caller names with
-o/--out, created if it is not there yet — the document is the user's
artifact, not the project's data:

    proof.tex                      the complete DAG
    <user_id>_proof.tex            a user's lemmas
    <user_id>--<run_id>--<hex>_proof.tex  a lemma and everything it cites

with path-hostile characters becoming underscores. A re-run overwrites the
file: the document is a function of the files, so nothing in an older one
is worth preserving.

This module renders from the data its caller hands it — an ordered list of
lemma ids, their nodes with their proofs, the reference collection — and
writes one file; export/command.py's export_entry loads the DAG.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = ["name_for", "render", "tex_escape", "write"]

# A filename piece: the user_id and lemma_id go into the name, and a
# lemma_id is model-generated, so whatever the model makes of an id must
# survive in a filename. Path-hostile characters become underscores, the
# refutations.py way (refutations keep the un-abbreviated values in the
# file's fields; here the document's headings carry them).
_NAME_HOSTILE = re.compile(r"[^A-Za-z0-9._-]+")

# The ten LaTeX-special characters, for the identifiers the document puts
# in \texttt and its headings. Statement and proof text is never put
# through this: it is math-bearing, model-written LaTeX, and escaping it
# would corrupt it.
_TEX_SPECIALS = str.maketrans(
    {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
)


def tex_escape(text: Any) -> str:
    """An identifier (a user_id, a lemma_id, a reference id) made safe for
    LaTeX text mode: the ten special characters spelled out. Math-bearing
    content (statements, proofs, formal statements) never goes through
    this — see the module docstring."""
    return str(text).translate(_TEX_SPECIALS)


def _safe(part: Any) -> str:
    """A value as a filename piece: path-hostile characters (and runs of
    them) become underscores, and an empty result becomes "_" rather than
    leaving two dashes next to each other in the name."""
    return _NAME_HOSTILE.sub("_", str(part)).strip(".") or "_"


def name_for(
    owner: Optional[str] = None, lemma_id: Optional[str] = None
) -> str:
    """The export file's name, in the directory -o/--out names: proof.tex for
    the complete DAG, <user_id>_proof.tex for a user's lemmas,
    <lemma id, slashes as -->_proof.tex for a lemma and its dependencies."""
    if lemma_id is not None:
        return "--".join(_safe(part) for part in lemma_id.split("/")) + "_proof.tex"
    if owner is not None:
        return f"{_safe(owner)}_proof.tex"
    return "proof.tex"


def _citations(node: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """A node's citations: the cited lemma ids and the cited reference ids,
    duplicates collapsed, the record's order kept."""
    lemma_ids: List[str] = []
    refs: List[str] = []
    for field, out in (("cited_lemmas", lemma_ids), ("cited_references", refs)):
        for entry in node.get(field, []):
            if isinstance(entry, str) and entry.strip() and entry not in out:
                out.append(entry)
    return lemma_ids, refs


def render(
    *,
    conjecture_name: str,
    scope: str,
    order: Sequence[str],
    lemmas: Mapping[Any, Mapping[str, Any]],
    references: Optional[Sequence[Mapping[str, Any]]] = None,
    conjecture: str = "",
    date: Optional[str] = None,
    computations: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    """The LaTeX document of the given lemmas, in the given order.

    computations maps a script's hash to what the appendix shows of it —
    "lang", "produced_output", "script" (its text, None when no intact
    copy is on disk) and "ran_with" (the tools recorded running it). A
    lemma that rests on computations points to them, and the appendix
    lists every one the document's lemmas cite, script verbatim.

    order is the lemmas in dependency order (dependencies first) — the
    order the caller settled, the way _citation_closure or
    _dependency_order in core/dag.py settles it — and the document numbers
    its sections in that order, so "Lemma 3" always stands on "Lemma 1"
    and "Lemma 2" when they are its dependencies. lemmas maps each lemma
    id in order to its node, proof included; a citation of a lemma outside
    the document (another user's, in a single-user export) is shown by its
    id. references is the reference collection (references.md's objects).
    An id in order that lemmas does not hold is skipped rather than
    rendered empty.
    """
    position: Dict[str, int] = {}
    body: List[str] = []
    cited_code: List[str] = []
    for i, lid in enumerate(order, 1):
        node = lemmas.get(lid)
        if not isinstance(node, Mapping):
            continue
        position[lid] = i
        body.append(f"\\section*{{Lemma {i}: \\texttt{{{tex_escape(lid)}}}}}")
        # Steps the counter the heading's number names, so \label below
        # is a real target and \ref{lem:..} prints this lemma's number.
        body.append("\\refstepcounter{lem}")
        body.append(f"\\label{{lem:{i}}}")
        body.append("")
        statement = str(node.get("statement") or "").strip()
        proof = str(node.get("proof") or "").strip()
        if statement:
            body.append("\\begin{lemma}")
            body.append(statement)
            body.append("\\end{lemma}")
            body.append("")
        if proof:
            body.append("\\begin{proof}")
            body.append(proof)
            body.append("\\end{proof}")
            body.append("")
        cited_ids, cited_refs = _citations(node)
        cites: List[str] = []
        for cited in cited_ids:
            if cited in position:
                cites.append(f"Lemma~\\ref{{lem:{position[cited]}}}")
            else:
                cites.append(f"\\texttt{{{tex_escape(cited)}}}")
        cites.extend(
            f"reference \\texttt{{{tex_escape(rid)}}}" for rid in cited_refs
        )
        if cites:
            body.append("\\textit{Cites:} " + ", ".join(cites) + ".")
        code_ids = [str(e.get("hash") or "") for e in node.get("code") or []
                    if isinstance(e, Mapping) and e.get("hash")]
        if code_ids:
            for h in code_ids:
                if h not in cited_code:
                    cited_code.append(h)
            body.append(
                "\\textit{Rests on computations:} "
                + ", ".join(f"\\texttt{{[code:{h[:8]}]}} "
                            f"(\\ref{{code:{h[:8]}}})" for h in code_ids)
                + "."
            )
        body.append("")

    # The References section: the references the document's lemmas cite,
    # in first-citation order, each with its slogan, formal statement and
    # source. An id references.md does not hold is listed anyway, marked
    # as such: the document should not pretend a citation resolves.
    refs_by_id: Dict[str, Mapping[str, Any]] = {}
    for ref in references or []:
        if isinstance(ref, Mapping):
            rid = str(ref.get("id") or "").strip()
            if rid:
                refs_by_id[rid] = ref
    cited_ref_ids: List[str] = []
    for lid in order:
        node = lemmas.get(lid)
        if not isinstance(node, Mapping):
            continue
        for rid in _citations(node)[1]:
            if rid not in cited_ref_ids:
                cited_ref_ids.append(rid)

    doc: List[str] = [
        "\\documentclass[11pt]{article}",
        "\\usepackage{amsmath}",
        "\\usepackage{amssymb}",
        "\\usepackage{amsthm}",
        "\\usepackage[margin=2.5cm]{geometry}",
        "",
        "\\newtheorem{lemma}{Lemma}",
        # A counter the lemma sections step so their \label has a real
        # target: a \section* is unnumbered, so on its own it gives a
        # following \label nothing to point at, and every \ref to the
        # section would come out undefined.
        "\\newcounter{lem}",
        "\\newcounter{comp}",
        "",
        f"\\title{{{tex_escape(conjecture_name)}}}",
        f"\\author{{{tex_escape(scope)}}}",
        f"\\date{{{date or datetime.now(timezone.utc).strftime('%Y-%m-%d')}}}",
        "",
        "\\begin{document}",
        "\\maketitle",
    ]
    if conjecture.strip():
        doc.append("")
        doc.append("\\section*{The conjecture}")
        doc.append(conjecture.strip())
    if body:
        doc.append("")
        doc.extend(body)
    if cited_ref_ids:
        doc.append("")
        doc.append("\\section*{References}")
        doc.append("\\begin{description}")
        for rid in cited_ref_ids:
            doc.append(f"\\item[\\texttt{{{tex_escape(rid)}}}]")
            ref = refs_by_id.get(rid)
            if ref is None:
                doc.append("(not in references.md)")
                continue
            slogan = str(ref.get("slogan") or "").strip()
            formal = str(ref.get("formal statement") or "").strip()
            source = str(ref.get("reference") or "").strip()
            if slogan:
                doc.append(f"\\textit{{{slogan}}}")
            if formal:
                doc.append(formal)
            if source:
                doc.append(f"\\textit{{(source: {tex_escape(source)})}}")
        doc.append("\\end{description}")
    if cited_code:
        # The computations: every script the document's lemmas rest on,
        # exactly as it ran, with the RESULT recorded from it. A script is
        # shown verbatim, so a line that would end the environment early
        # is broken apart.
        doc.append("")
        doc.append("\\section*{Computations}")
        for i, h in enumerate(cited_code, 1):
            c = (computations or {}).get(h) or {}
            doc.append(f"\\subsection*{{Computation {i}: "
                       f"\\texttt{{[code:{h[:8]}]}}}}")
            doc.append("\\refstepcounter{comp}")
            doc.append(f"\\label{{code:{h[:8]}}}")
            ran = ", ".join(c.get("ran_with") or []) or "no recorded execution"
            doc.append(f"Language \\texttt{{{tex_escape(c.get('lang') or '')}}}; "
                       f"run with {tex_escape(ran)}; script SHA-256 "
                       f"\\texttt{{{h}}}.")
            doc.append("")
            doc.append("Recorded \\texttt{RESULT}:")
            doc.append("\\begin{verbatim}")
            doc.append(json.dumps(c.get("produced_output"), sort_keys=True, indent=1))
            doc.append("\\end{verbatim}")
            script = c.get("script")
            if script is None:
                doc.append("\\textit{(No intact copy of this script is on disk.)}")
            else:
                doc.append("\\begin{verbatim}")
                doc.append(script.rstrip("\n").replace("\\end{verbatim}",
                                                     "\\end {verbatim}"))
                doc.append("\\end{verbatim}")
            doc.append("")
    doc.append("")
    doc.append("\\end{document}")
    return "\n".join(doc) + "\n"


def write(path: Union[str, os.PathLike], text: str) -> Path:
    """Write the document, the way the refutations are
    written: a temp file beside it, then os.replace, so a crashed write
    leaves either the old document or the new one, never a half of both.
    Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, p)
    return p
