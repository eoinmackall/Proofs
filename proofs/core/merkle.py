"""Merkle hashes of the DAG's lemmas.

Every text is hashed once, raw, and never inside JSON:

    statement_hash = sha256(normalized statement)
    proof_hash     = sha256(normalized proof)
    lemma_hash     = sha256(canonical JSON of
                            {"statement": statement_hash,
                             "proof":     proof_hash,
                             "cited":     [sorted statement hashes of
                                           everything it cites],
                             "code":      [its computations]})

A cited lemma and a cited reference contribute the same thing, the hash of
their statement (a reference's "formal statement"): a citation pins the
result, not its label, its slogan or how it was proved. A citation that
resolves to nothing contributes a marker (_unresolved) that names it, so
the hash still exists and moves when the citation is fixed. "code" is the
lemma's computations (code_entries), [] for a lemma with none: a different
script, or the same script claimed to give a different result, is a
different proof.

The hash therefore covers exactly what a verifier is shown: the lemma's own
statement and proof and the statements of what it cites. A new statement for
a lemma changes the hash of every lemma citing it, and those must be
verified again; a new proof under the same statement changes only its own.
Whether the ground a lemma stands on is sound is not the hash's question but
suspension's (suspension.py).

Hashing the texts raw is what keeps a build of the whole DAG fast: escaping
megabytes of proof into JSON costs more than everything else together. And
because a lemma hash needs only the two text hashes of the lemma itself and
the statement hashes of what it cites, a DAG that keeps those (core/dag.py)
recomputes any lemma's hash without reading a proof.

No hash is stored in a stream: it is a function of the files' contents,
recomputed when the DAG is built. That is what makes it shared — every user
who builds the same files computes the same hashes — and how a certificate or
refutation written by one user stays checkable by every other.

"Normalized" is a fixed whitespace canonical form (normalize()): no
surrounding whitespace, line endings to \\n, no trailing line whitespace, no
blank-line runs. Text that differs only in those cosmetics hashes alike; a
change to any word, symbol or line break does not. A stream stores the
statement and proof already normalized (the writer normalizes once, at
commit), so the load hashes them as stored (normalized_hash) and never
normalizes; normalizing is idempotent, so the hashes are the same either way.

No lemma's hash depends on another lemma's hash — a cited lemma contributes
only its statement — so every hash is computed on its own, in any order,
and a commit hashes its one lemma without walking anything below it. A
citation cycle is the one thing that needs the whole graph: Merkle.all()
finds the cycles in one linear pass, and a lemma on a cycle, or citing its
way up to one, has no hash, so no certificate can be valid for it. That
pass runs when the DAG is loaded; it is the safety net for what a run
cannot cause (a hand edit, a git restore, repairs on two machines).

A version hash (version_hash) names one version of a lemma by its own
content — statement, proof, the ids it cites, its computations — and not by
the statements of what it cites, so it does not move when a cited lemma
changes: it is what a reproof's "supersedes" names (core/dag.py).

This module imports nothing from this package.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

__all__ = [
    "Merkle",
    "MerkleCycleError",
    "canonical_json",
    "code_entries",
    "lemma_hash",
    "normalize",
    "normalized_hash",
    "proof_hash",
    "reference_hash",
    "sha256_hex",
    "statement_hash",
    "version_hash",
]


class MerkleCycleError(Exception):
    """A citation cycle: these lemmas' hashes depend on each other, so none
    of them can be computed in dependency order."""


# ----------------------------------------------------------------------------
# Canonical forms
# ----------------------------------------------------------------------------
def normalize(text: Any) -> str:
    """The canonical whitespace form of a statement or proof.

    The surrounding whitespace goes, line endings become \\n, trailing
    whitespace is dropped from every line, and runs of blank lines collapse
    to a single blank line. A non-string reads as itself: None becomes "",
    anything else str()."""
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    lines = [
        line.rstrip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    ]
    out: List[str] = []
    for line in lines:
        if line == "" and out and out[-1] == "":
            continue
        out.append(line)
    return "\n".join(out)


def canonical_json(obj: Any) -> str:
    """Sorted keys, compact separators, non-ASCII kept as UTF-8."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_hash(text: str) -> str:
    """The hash of text already in normal form, as a stream stores it: the
    same as statement_hash or proof_hash of it, without normalizing again."""
    return sha256_hex(text.encode("utf-8"))


def statement_hash(statement: Any) -> str:
    """The hash of a normalized statement: what a cited lemma or reference
    contributes to the hash of a lemma citing it."""
    return normalized_hash(normalize(statement))


def proof_hash(proof: Any) -> str:
    """The hash of a normalized proof."""
    return normalized_hash(normalize(proof))


def reference_hash(reference: Mapping[str, Any]) -> str:
    """A cited reference's contribution: its formal statement's hash."""
    return statement_hash(reference.get("formal statement", ""))


def code_entries(node: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The computations a lemma rests on, as the hash takes them: each
    script's hash, language and RESULT, sorted by script hash."""
    raw = node.get("code")
    if not isinstance(raw, list):
        return []
    out = [
        {
            "hash": str(e.get("hash") or ""),
            "lang": str(e.get("lang") or ""),
            "produced_output": e.get("produced_output"),
        }
        for e in raw
        if isinstance(e, Mapping)
    ]
    return sorted(out, key=lambda e: (e["hash"], e["lang"]))


def _unresolved(kind: str, ident: str) -> str:
    """What a citation that resolves to nothing contributes: a marker that
    names it, the same for every user, and different once it is fixed."""
    marker = {"unresolved_citation": {"kind": kind, "id": ident}}
    return sha256_hex(canonical_json(marker).encode("utf-8"))


def lemma_hash(statement_h: str, proof_h: str, cited: Iterable[str],
               code: List[Dict[str, Any]]) -> str:
    """The lemma hash from its parts (see the module docstring)."""
    return sha256_hex(canonical_json({
        "statement": statement_h,
        "proof": proof_h,
        "cited": sorted(set(cited)),
        "code": code,
    }).encode("utf-8"))


def text_hashes(node: Mapping[str, Any]) -> Tuple[str, str]:
    """(statement_hash, proof_hash) of a node: the ones it carries (an
    in-memory DAG node, which keeps no proof), or computed from its text."""
    sh = node.get("statement_hash") or statement_hash(node.get("statement"))
    ph = node.get("proof_hash") or proof_hash(node.get("proof"))
    return sh, ph


def ids(raw: Any) -> List[str]:
    """The ids a citation list holds: its non-empty strings, stripped."""
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if isinstance(x, str) and x.strip()]


def version_hash(node: Mapping[str, Any]) -> str:
    """One version of a lemma, by its own content: statement, proof, the
    ids it cites, its computations."""
    sh, ph = text_hashes(node)
    return sha256_hex(canonical_json({
        "statement": sh,
        "proof": ph,
        "cited_lemmas": sorted(set(ids(node.get("cited_lemmas")))),
        "cited_references": sorted(set(ids(node.get("cited_references")))),
        "code": code_entries(node),
    }).encode("utf-8"))


# ----------------------------------------------------------------------------
# The Merkle hashes of one complete DAG
# ----------------------------------------------------------------------------
class Merkle:
    """The hashes of one complete DAG.

    lemmas maps a lemma id to its node: statement (or statement_hash),
    proof (or proof_hash), cited_lemmas (ids), cited_references (ids) and
    code. references is the reference collection (references.md).

    clean says the nodes are already in the in-memory DAG's form
    (core/dag.py): statement_hash and proof_hash present, cited_lemmas and
    cited_references lists of ids as ids() makes them, code as
    code_entries() makes it — so they are used as they are, not cleaned
    again. A node given to node_hash is cleaned unless said otherwise."""

    def __init__(
        self,
        lemmas: Mapping[str, Mapping[str, Any]],
        references: Optional[Iterable[Mapping[str, Any]]] = None,
        clean: bool = False,
    ) -> None:
        self._lemmas = lemmas
        self._clean = clean
        self._ref_hashes: Dict[str, str] = {}
        for ref in references or []:
            if not isinstance(ref, Mapping):
                continue
            rid = str(ref.get("id") or "").strip()
            if rid:
                self._ref_hashes[rid] = reference_hash(ref)

    def node_hash(self, node: Mapping[str, Any], clean: bool = False) -> str:
        """The hash node has over this DAG's statements and references,
        with no cycle check: what it cites contributes statement hashes
        only, so nothing below it is visited. clean: node is in the
        in-memory DAG's form (see the class docstring)."""
        if clean:
            return lemma_hash(node["statement_hash"], node["proof_hash"],
                              self._cited_hashes(node, True), node["code"])
        sh, ph = text_hashes(node)
        return lemma_hash(sh, ph, self._cited_hashes(node), code_entries(node))

    def all(self) -> Dict[str, Optional[str]]:
        """Every lemma's hash, None for a lemma on a citation cycle or
        citing its way up to one."""
        blocked = self.blocked()
        clean = self._clean
        return {
            lid: None if lid in blocked else self.node_hash(node, clean)
            for lid, node in self._lemmas.items()
        }

    def blocked(self) -> Set[str]:
        """The lemmas on a citation cycle, and those citing their way up to
        one: Tarjan's strongly connected components, iteratively, in one
        pass. A component is finished only after every component it cites,
        so whether it reaches a cycle is known when it is."""
        lemmas = self._lemmas
        cited = ((lambda node: node["cited_lemmas"]) if self._clean
                 else (lambda node: ids(node.get("cited_lemmas"))))
        edges = {
            lid: [c for c in cited(node) if c in lemmas]
            for lid, node in lemmas.items()
        }
        index: Dict[str, int] = {}
        low: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: Set[str] = set()
        blocked: Set[str] = set()
        for root in lemmas:
            if root in index:
                continue
            index[root] = low[root] = len(index)
            stack.append(root)
            on_stack.add(root)
            work = [(root, iter(edges[root]))]
            while work:
                v, it = work[-1]
                for w in it:
                    if w not in index:
                        index[w] = low[w] = len(index)
                        stack.append(w)
                        on_stack.add(w)
                        work.append((w, iter(edges[w])))
                        break
                    if w in on_stack:
                        low[v] = min(low[v], index[w])
                else:
                    work.pop()
                    if work:
                        u = work[-1][0]
                        low[u] = min(low[u], low[v])
                    if low[v] == index[v]:
                        component: List[str] = []
                        while True:
                            w = stack.pop()
                            on_stack.discard(w)
                            component.append(w)
                            if w == v:
                                break
                        if (len(component) > 1 or v in edges[v]
                                or any(w in blocked for m in component
                                       for w in edges[m])):
                            blocked.update(component)
        return blocked

    def _cited_hashes(self, node: Mapping[str, Any], clean: bool = False) -> List[str]:
        hashes: List[str] = []
        lemmas = self._lemmas
        for lid in node["cited_lemmas"] if clean else ids(node.get("cited_lemmas")):
            cited = lemmas.get(lid)
            if cited is None:
                hashes.append(_unresolved("lemma", lid))
            elif self._clean:
                hashes.append(cited["statement_hash"])
            else:
                hashes.append(text_hashes(cited)[0])
        for rid in node["cited_references"] if clean else ids(node.get("cited_references")):
            hashes.append(self._ref_hashes.get(rid) or _unresolved("reference", rid))
        return hashes
