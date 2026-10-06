"""The LaTeX export of a DAG.

`proofs export PATH [lemma_id]` writes a LaTeX document of the DAG, in
dependency order: every lemma comes after the lemmas it cites, so the
document reads as a sequence of results each resting on the ones above
it. What PATH names is what the document holds:

    a user's DAG file, no lemma id   -> that user's lemmas
    a user's DAG file, with lemma id -> that lemma and everything it cites
    the conjecture directory         -> the complete DAG

The document's body is the lemmas' own text: each statement in a lemma
environment, each proof in a proof environment, written verbatim. The
model writes statements and proofs as LaTeX text and the Merkle hash
covers them as-is, so nothing here rewrites or escapes them — a lemma id
or user_id in a heading is escaped, since an id is an identifier, not
math. A lemma's citations are listed under it: a cited lemma that is in
the document is cross-referenced to its section, a citation the document
does not hold (another user's lemma in a single-user export, or a
dangling citation) is shown by its pair, and the references the document's
lemmas cite are collected in a References section at the end, each with
its slogan, formal statement and source.

The output file is written into the directory the caller names with
-o/--out, created if it is not there yet — the document is the
user's artifact, not the project's data, so it never lands in the
project on its own:

    proof.tex                        the complete DAG
    <user_id>_proof.tex              a user's lemmas
    <user_id>--<lemma_id>_proof.tex  a lemma and everything it cites

with path-hostile characters in the ids becoming underscores, the way
refutations.py names its files. A re-run overwrites the file: the
document is a function of the DAG files and references.md, the way a
Merkle hash is, so nothing in an older one is worth preserving.

This module is deliberately standalone, the way merkle.py and
refutations.py are: it renders from the data its caller hands it — an
ordered list of (user_id, lemma_id) pairs, the complete DAG's lemmas
mapping, the reference collection — and writes one file. main.py's
export_entry is the one that loads the DAG and knows what PATH named.
"""

from __future__ import annotations

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
    <user_id>--<lemma_id>_proof.tex for a lemma and its dependencies —
    the same three-way split as what PATH names (see the module
    docstring)."""
    if owner is None:
        return "proof.tex"
    if lemma_id is None:
        return f"{_safe(owner)}_proof.tex"
    return f"{_safe(owner)}--{_safe(lemma_id)}_proof.tex"


def _pair_of(entry: Any) -> Optional[Tuple[str, str]]:
    """The (user_id, lemma_id) pair a citation entry names, or None.

    The shapes a lemma citation takes: the in-memory tuple (load_dag's
    view) and the stored {"user_id", "lemma_id"} object (the DAG file's
    and the prover's spelling). A bare id is not a pair: a lemma_id is
    unique only within one user's file, so on its own it names no
    particular lemma."""
    if isinstance(entry, Mapping):
        uid = entry.get("user_id")
        lid = entry.get("lemma_id")
        if isinstance(uid, str) and isinstance(lid, str) and uid and lid:
            return (uid, lid)
    elif isinstance(entry, (tuple, list)) and len(entry) == 2:
        uid, lid = entry
        if isinstance(uid, str) and isinstance(lid, str) and uid and lid:
            return (uid, lid)
    return None


def _citations(
    node: Mapping[str, Any]
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """A node's citations in the two stored fields' shapes: the cited
    lemma pairs and the cited reference ids, duplicates collapsed, the
    file's order kept (see main.load_dag())."""
    pairs: List[Tuple[str, str]] = []
    refs: List[str] = []
    seen_pairs: set = set()
    seen_refs: set = set()
    for entry in node.get("cited_lemmas", []):
        pair = _pair_of(entry)
        if pair is not None and pair not in seen_pairs:
            seen_pairs.add(pair)
            pairs.append(pair)
    for entry in node.get("cited_references", []):
        if isinstance(entry, str) and entry.strip() and entry not in seen_refs:
            seen_refs.add(entry)
            refs.append(entry)
    return pairs, refs


def render(
    *,
    conjecture_name: str,
    scope: str,
    order: Sequence[Tuple[str, str]],
    lemmas: Mapping[Any, Mapping[str, Any]],
    references: Optional[Sequence[Mapping[str, Any]]] = None,
    conjecture: str = "",
    date: Optional[str] = None,
) -> str:
    """The LaTeX document of the given lemmas, in the given order.

    order is the lemmas in dependency order (dependencies first) — the
    order the caller settled, the way _citation_closure or
    _dependency_order in main.py settles it — and the document numbers
    its sections in that order, so "Lemma 3" always stands on "Lemma 1"
    and "Lemma 2" when they are its dependencies. lemmas is the complete
    DAG's mapping from (user_id, lemma_id) to node, so a citation to a
    lemma outside the document (another user's, in a single-user export)
    can still be shown; references is the reference collection
    (references.md's objects). A pair in order that lemmas does not hold
    is skipped rather than rendered empty.
    """
    position: Dict[Tuple[str, str], int] = {}
    body: List[str] = []
    for i, pair in enumerate(order, 1):
        node = lemmas.get(pair)
        if not isinstance(node, Mapping):
            continue
        position[pair] = i
        u, lid = pair
        body.append(f"\\section*{{Lemma {i}: \\texttt{{{tex_escape(lid)}}} "
                    f"(\\texttt{{{tex_escape(u)}}})}}")
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
        cited_pairs, cited_refs = _citations(node)
        cites: List[str] = []
        for pair_cited in cited_pairs:
            if pair_cited in position:
                cites.append(f"Lemma~\\ref{{lem:{position[pair_cited]}}}")
            else:
                cites.append(
                    f"\\texttt{{{tex_escape(pair_cited[0])}:{tex_escape(pair_cited[1])}}}"
                )
        cites.extend(
            f"reference \\texttt{{{tex_escape(rid)}}}" for rid in cited_refs
        )
        if cites:
            body.append("\\textit{Cites:} " + ", ".join(cites) + ".")
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
    for pair in order:
        node = lemmas.get(pair)
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
    doc.append("")
    doc.append("\\end{document}")
    return "\n".join(doc) + "\n"


def write(path: Union[str, os.PathLike], text: str) -> Path:
    """Write the document, the way the DAG files and refutations are
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
