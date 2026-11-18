"""The DAG: every run's lemma records, read into one in-memory view, and the
commits that add to it.

A lemma is a record in the dags/ stream of the run that proved it
(streams.py):

    conjectures/NAME/dags/<user_id>/<run_id>/000001.jsonl

    {"type": "lemma",
     "id": "<user_id>/<run_id>/<6 hex digits>",
     "statement": "...",
     "proof": "...",
     "cited_lemmas": ["<lemma id>", ...],
     "cited_references": ["<reference id>", ...],
     "code": [{"hash", "lang", "produced_output"}, ...],
     "proved_at": "<ISO 8601 UTC>",
     "model": "<the model that wrote the proof>"}     when known

The statement and proof are stored normalized (merkle.normalize, applied
once by the writer), and the load hashes them as they are stored.

Its id is drawn when the candidate is created (new_lemma_id) — the user,
the run, and six random hex digits, drawn again if the run has used them —
so it is unique everywhere without any coordination, and a lemma is known by
the same id from the moment it is proposed to the moment it is committed.
proofs repair does not touch the original: it appends a reproof to its own
run's stream, the whole new version under the same id,

    {"type": "reproof", "id": "<the lemma's id>",
     "supersedes": "<version hash of the version it replaces>", ...}

and the current version of a lemma is the version no other version
supersedes, whatever order the lines are in. supersedes names a version by
its own content (merkle.version_hash), which does not move when something it
cites changes. proofs prune compacts a chain (core/prune.py): the current
reproof is rewritten in place as a plain lemma record and every other
version is deleted, so the reader meets a repaired lemma as an ordinary one,
in the repairer's stream.

More than one version that nothing supersedes is a fork: two repairs of one
version, or a reproof of a version a prune has since deleted (a repair that
raced the prune), against the current one. Every reader decides a fork the
same way (_resolve):

  * the versions all state the same thing (up to whitespace): the lemma is
    valid if any of them is certified, and the current version is a
    certified one (the smaller lemma hash, if several; the smaller hash of
    all of them, if none is). proofs prune keeps that one, when it is
    certified, and deletes the rest. Nothing citing the lemma moves, since a
    citation pins only the statement.
  * the statements differ: the lemma is suspended (Dag.conflicts), and
    reported on every load until a person deletes all but one version.
    proofs prune leaves it alone.

Every command builds the DAG once, when it starts (Dag.load), and works from
it and its own commits: what another process writes in the meantime is not
seen. The in-memory DAG keeps every lemma's current version without its
proof — the statement and its hash, the proof's hash, the citations, the
computations, who wrote it, where — and every lemma's Merkle hash, computed
at load; a proof is read back from its stream file when one is needed. Who
cites each lemma, each lemma's depth and the sketches of the lemmas'
ancestries (citations.py) are built from them the first time they are
asked for, and carried across the command's commits.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..loops.json_repair import _FENCE_RE
from ..loops.reasoning import _recorded_model
from ..tools.run_code import computations
from ..workspace import config
from . import certificates, citations, merkle, refutations, resolution, streams, suspension

DIR_NAME = "dags"
LEMMA = "lemma"
REPROOF = "reproof"

# What a lemma version keeps in memory besides its statement and citations.
_KEPT = ("statement", "cited_lemmas", "cited_references", "code", "proved_at")


def load_file(filepath: Any) -> str:
    """Read prompt or markdown file contents. Accepts str or Path."""
    if not filepath or not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def owner_of(lemma_id: str) -> str:
    """The user whose run proved a lemma: its id's first part."""
    return str(lemma_id).split("/", 1)[0]


def _lemma_id_from(entry: Any) -> Optional[str]:
    """The lemma id a citation entry declares — a string id, or an object
    with an "id" — or None."""
    if isinstance(entry, dict):
        entry = entry.get("id")
    if isinstance(entry, str) and entry.strip():
        return entry.strip()
    return None


def _citation_display(entry: Any) -> str:
    """How a citation entry reads in a log line."""
    lid = _lemma_id_from(entry)
    if lid is not None:
        return lid
    try:
        return json.dumps(entry, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(entry)


# ----------------------------------------------------------------------------
# The in-memory DAG
# ----------------------------------------------------------------------------
def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _version(record: Dict[str, Any], path: Path, line: int, prover: str,
             run_id: str, root: Path, file: Optional[str] = None) -> Dict[str, Any]:
    """One lemma version as memory keeps it: the record without its proof,
    with the hashes of its texts and where it is (file, path relative to
    root, when the caller already has it). Its version hash is computed
    only when something needs it (version_of)."""
    node = {k: record.get(k) for k in _KEPT}
    node["cited_lemmas"] = merkle.ids(record.get("cited_lemmas"))
    node["cited_references"] = merkle.ids(record.get("cited_references"))
    node["code"] = merkle.code_entries(record)
    node["statement"] = str(record.get("statement") or "")
    node["statement_hash"] = merkle.normalized_hash(node["statement"])
    node["proof_hash"] = merkle.normalized_hash(str(record.get("proof") or ""))
    node["id"] = str(record.get("id") or "")
    node["type"] = record.get("type")
    node["supersedes"] = record.get("supersedes")
    node["prover"] = prover
    node["run_id"] = run_id
    node["file"] = file if file is not None else _relative(path, root)
    node["line"] = line
    return node


def version_of(node: Dict[str, Any]) -> str:
    """The version hash of an in-memory version (merkle.version_hash), from
    the text hashes it already holds; computed on first use and kept on the
    node. Only a lemma with more than one record, or a reproof, needs one
    at load; a commit of a reproof needs its current version's."""
    v = node.get("version")
    if v is None:
        v = node["version"] = merkle.version_hash(node)
    return v


def _leaves(versions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The versions of one lemma that no other version supersedes — its
    candidates for current — one per version hash (the first record of
    each), in load order. Every version, should supersedes run in a circle."""
    unique: Dict[str, Dict[str, Any]] = {}
    for v in versions:
        unique.setdefault(version_of(v), v)
    superseded = {
        v.get("supersedes") for v in unique.values()
        if v["type"] == REPROOF and v.get("supersedes") != v["version"]
    }
    leaves = [v for k, v in unique.items() if k not in superseded]
    return leaves or list(unique.values())


def _resolve(
    versions: Dict[str, List[Dict[str, Any]]],
    references: List[Dict[str, Any]],
    certs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]],
           Dict[str, List[Dict[str, Any]]]]:
    """Every lemma's current version (see the module docstring on forks):
    (lemmas, {forked id: where its versions are}, {conflicted id: the
    same}, {id of a fork whose versions agree: its versions, best first}).

    A forked lemma whose versions state different things takes the version
    with the smaller version hash, so the statement its citers are hashed
    against is the same for every reader; it is suspended all the same.
    Every other statement is then fixed, so each version of a fork whose
    versions agree can be hashed on its own, and the certified one chosen.
    A version that turns out to have no hash (a citation cycle) cannot be
    certified, so it never wins over one that has a hash: the load's cycle
    pass moves such a fork on to its next version (_settle)."""
    lemmas: Dict[str, Dict[str, Any]] = {}
    agreeing: Dict[str, List[Dict[str, Any]]] = {}
    forked: Dict[str, List[str]] = {}
    conflicts: Dict[str, List[str]] = {}
    for lid, vs in versions.items():
        if len(vs) == 1 and not vs[0].get("supersedes"):
            lemmas[lid] = vs[0]         # one record, nothing to decide
            continue
        leaves = _leaves(vs)
        lemmas[lid] = min(leaves, key=version_of)
        if len(leaves) == 1:
            continue
        forked[lid] = sorted({f"{v['file']}:{v['line']}" for v in leaves})
        if len({v["statement_hash"] for v in leaves}) == 1:
            agreeing[lid] = leaves
        else:
            conflicts[lid] = forked[lid]
    ranked: Dict[str, List[Dict[str, Any]]] = {}
    if not agreeing:
        return lemmas, forked, conflicts, ranked
    tree = merkle.Merkle(lemmas, references, clean=True)
    certs_of: Dict[str, List[Dict[str, Any]]] = {}
    for c in certs:
        if c.get("lemma_id") in agreeing:
            certs_of.setdefault(c["lemma_id"], []).append(c)
    for lid, leaves in agreeing.items():
        scored = []
        for v in leaves:
            h = tree.node_hash(v, clean=True)
            valid = any(certificates.is_valid(c, h) for c in certs_of.get(lid, []))
            scored.append((not valid, h, version_of(v), v))
        ranked[lid] = [s[3] for s in sorted(scored, key=lambda s: s[:3])]
        lemmas[lid] = ranked[lid][0]
    return lemmas, forked, conflicts, ranked


def _settle(lemmas: Dict[str, Dict[str, Any]],
            ranked: Dict[str, List[Dict[str, Any]]],
            references: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Every lemma's hash (Merkle.all), after moving each agreeing fork
    whose chosen version has none (a citation cycle) on to its next
    version, until each has a hash or no version is left to try. Another
    cycle pass follows each round of moves; without a cycle there is one
    pass and no round."""
    tree = merkle.Merkle(lemmas, references, clean=True)
    hashes = tree.all()
    tried = {lid: 0 for lid in ranked}
    while True:
        moved = False
        for lid, versions in ranked.items():
            if hashes[lid] is None and tried[lid] + 1 < len(versions):
                tried[lid] += 1
                lemmas[lid] = versions[tried[lid]]
                moved = True
        if not moved:
            return hashes
        hashes = tree.all()


class Dag:
    """The in-memory DAG a command works from: built once, from the streams
    as they stand when it starts (Dag.load), and changed only by the
    command's own commits.

    lemmas maps each lemma id to its current version (see _version);
    hashes each id to its Merkle hash (None for a lemma of a citation
    cycle); references is references.md; certificates every certificate
    record, and refutations every refutation file as (path, record). The
    suspension is computed from them (suspension()), and so is who cites
    each lemma (cited_by_map()). warnings are what the load found worth
    saying once (forks).

    A Dag is never changed in place: the with_* methods return a new one,
    and the command's context swaps it in (Context.dag) under the DAG lock.
    A reader in another thread keeps the one it holds, whole.
    """

    def __init__(
        self,
        lemmas: Dict[str, Dict[str, Any]],
        hashes: Dict[str, Optional[str]],
        references: List[Dict[str, Any]],
        certificate_lines: List[Dict[str, Any]],
        refutation_files: List[Tuple[Path, Dict[str, Any]]],
        root: Path,
        warnings: Optional[List[str]] = None,
        forked: Optional[Dict[str, List[str]]] = None,
        conflicts: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.lemmas = lemmas
        self.hashes = hashes
        self.references = references
        self.certificates = certificate_lines
        self.refutations = refutation_files
        self.root = root
        self.warnings = warnings or []
        # Forks (see the module docstring): every forked lemma, with where
        # its candidate versions are (file:line), and those of them whose
        # versions state different things.
        self.forked: Dict[str, List[str]] = dict(forked or {})
        self.conflicts: Dict[str, List[str]] = dict(conflicts or {})
        self._suspension: Optional[suspension.Suspension] = None
        self._tree: Optional[merkle.Merkle] = None
        self._cited_by: Optional[Dict[str, List[str]]] = None
        self._depth: Optional[Dict[str, int]] = None
        self._ancestry: Optional[citations.Ancestry] = None

    @classmethod
    def load(cls, ctx) -> "Dag":
        """Read every run's dags/ and certificates/ streams, the
        refutations and references.md, resolve each lemma's current
        version, and hash every lemma."""
        root = Path(ctx.paths.conjecture_root)
        versions: Dict[str, List[Dict[str, Any]]] = {}
        files: Dict[Path, str] = {}
        for user_id, run_id, path, line, rec in streams.records(root / DIR_NAME):
            if rec.get("type") not in (LEMMA, REPROOF):
                continue
            lid = str(rec.get("id") or "").strip()
            if not lid:
                raise streams.StreamError(path, line, "a lemma record with no id")
            if path not in files:
                files[path] = _relative(path, root)
            versions.setdefault(lid, []).append(
                _version(rec, path, line, user_id, run_id, root, files[path]))
        references = load_references(ctx)
        certs = certificates.load_all(root)
        refs = refutations.load_all(root)

        lemmas, forked, conflicts, ranked = _resolve(versions, references, certs)
        hashes = _settle(lemmas, ranked, references)
        # One line however many forks there are, naming one of them, with
        # where its versions are, for the person who has to settle it.
        warnings = []
        if conflicts:
            first = min(conflicts)
            warnings.append(
                f"{len(conflicts):,} lemma(s) have versions that state "
                f"different things, and are suspended until all but one "
                f"version of each is deleted by hand (e.g. {first}: "
                + ", ".join(conflicts[first]) + ")."
            )
        return cls(lemmas, hashes, references, certs, refs, root, warnings,
                   forked, conflicts)

    def _replace(self, susp: Optional["suspension.Suspension"] = None,
                 cited_by: Optional[Dict[str, List[str]]] = None,
                 depth: Optional[Dict[str, int]] = None,
                 ancestry: Optional["citations.Ancestry"] = None,
                 **changes: Any) -> "Dag":
        """A new Dag with changes; susp, when given, is its suspension —
        one the caller knows is still right — and otherwise it is computed
        when first asked for. What is read off the citations (who cites
        whom, depths, ancestries) is carried over when the lemmas are
        unchanged; when they change it is the caller's, for the new
        lemmas, or built when first asked for."""
        fields = dict(
            lemmas=self.lemmas, hashes=self.hashes, references=self.references,
            certificate_lines=self.certificates,
            refutation_files=self.refutations, root=self.root,
            warnings=self.warnings, forked=self.forked, conflicts=self.conflicts,
        )
        fields.update(changes)
        dag = Dag(**fields)
        dag._suspension = susp
        if "lemmas" in changes:
            dag._cited_by, dag._depth, dag._ancestry = cited_by, depth, ancestry
        else:
            dag._cited_by, dag._depth, dag._ancestry = (
                self._cited_by, self._depth, self._ancestry)
        return dag

    # -- reading -------------------------------------------------------------

    def suspension(self) -> "suspension.Suspension":
        """The suspension of this DAG (suspension.py), computed once."""
        if self._suspension is None:
            self._suspension = suspension.compute(
                self.lemmas, self.hashes, self.certificates,
                [ref for _path, ref in self.refutations], self.conflicts,
                citers=self.cited_by_map(),
            )
        return self._suspension

    def cited_by_map(self) -> Dict[str, List[str]]:
        """Every cited id, with the lemmas citing it (citations.py), built
        once. Shared with the Dags this one's commits make, so it is not
        to be changed."""
        if self._cited_by is None:
            self._cited_by = citations.build_cited_by(self.lemmas)
        return self._cited_by

    def cited_by(self, lemma_id: str) -> List[str]:
        """The lemmas that cite lemma_id directly, suspended or not; [] for
        a lemma nothing cites."""
        return list(self.cited_by_map().get(lemma_id, ()))

    def depths(self) -> Dict[str, int]:
        """Every lemma's depth (citations.depths), built once; a lemma of a
        citation cycle has none. Not to be changed."""
        if self._depth is None:
            self._depth = citations.depths(self.lemmas, self.cited_by_map())
        return self._depth

    def ancestry(self) -> "citations.Ancestry":
        """The sketches of every lemma's ancestry (citations.Ancestry),
        built once."""
        if self._ancestry is None:
            self._ancestry = citations.Ancestry.build(self.lemmas, self.cited_by_map())
        return self._ancestry

    def hash(self, lemma_id: str) -> str:
        """The lemma's Merkle hash. KeyError for a lemma the DAG does not
        hold; MerkleCycleError for one of a citation cycle."""
        if lemma_id not in self.lemmas:
            raise KeyError(f"{lemma_id!r} is not a lemma of this DAG")
        h = self.hashes.get(lemma_id)
        if h is None:
            raise merkle.MerkleCycleError(f"{lemma_id} is on a citation cycle")
        return h

    def hash_of(self, lemma_id: str, node: Dict[str, Any]) -> str:
        """The hash node would have under lemma_id, over this DAG and its
        references: a proof the command holds (its proof is in node), as
        the verifiers checked it or as it is about to be committed.

        No cycle check beyond what it cites directly: a new lemma is cited
        by nothing, and a reproof cannot cite the lemma it re-proves or
        anything above it (they are suspended, hidden from the prover). It
        does refuse to hash over a cited lemma that has no hash (a cycle
        the load found), as the load would."""
        for c in merkle.ids(node.get("cited_lemmas")):
            if c == lemma_id:
                raise merkle.MerkleCycleError(f"{lemma_id} cites itself")
            if c in self.lemmas and self.hashes.get(c) is None:
                raise merkle.MerkleCycleError(
                    f"{c}, which it cites, is on or above a citation cycle")
        return self._merkle().node_hash(node)

    def _merkle(self) -> "merkle.Merkle":
        if self._tree is None:
            self._tree = merkle.Merkle(self.lemmas, self.references, clean=True)
        return self._tree

    def refutations_for(self, lemma_id: str) -> List[Tuple[Path, Dict[str, Any]]]:
        """The lemma's refutations, as (path, record), oldest first."""
        return [
            (path, ref) for path, ref in self.refutations
            if str(ref.get("lemma_id") or "") == str(lemma_id)
        ]

    def proofs(self, lemma_ids: Iterable[str]) -> Dict[str, str]:
        """The proofs of the lemmas' current versions, read back from their
        stream files, each file read once."""
        by_file: Dict[str, List[str]] = {}
        for lid in lemma_ids:
            by_file.setdefault(self.lemmas[lid]["file"], []).append(lid)
        out: Dict[str, str] = {}
        for rel, lids in by_file.items():
            _header, recs = streams.read_file(self.root / rel)
            wanted = {(lid, self.lemmas[lid]["line"]) for lid in lids}
            for n, rec in recs:
                lid = str(rec.get("id") or "")
                if (lid, n) in wanted:
                    out[lid] = str(rec.get("proof") or "")
        for lid in lemma_ids:
            out.setdefault(lid, "")
        return out

    def proof(self, lemma_id: str) -> str:
        return self.proofs([lemma_id])[lemma_id]

    # -- the command's own changes ------------------------------------------

    def with_version(self, record: Dict[str, Any], path: Path, line: int,
                     prover: str, run_id: str) -> "Dag":
        """This DAG with a version just committed — a new lemma, or a
        reproof — as the current version of its id. The hashes that can
        move are recomputed, each on its own (merkle.py): the lemma's own,
        and those of the lemmas citing it if it changed its statement.

        There is no cycle check. A reproof cannot make a cycle: the prover
        cannot cite a suspended lemma, and the lemma it re-proves and
        everything citing it are suspended. One that breaks a cycle takes
        effect at the next load, whose cycle pass hashes the lemmas above
        it again; until then they keep no hash.

        A new lemma changes no other lemma's suspension — nothing cites it
        yet — so the suspension is carried over with the lemma's own status
        added (suspension.with_lemma), not recomputed. After a reproof it is
        recomputed when next asked for.

        Who cites whom, once built, is carried over: the lists of the
        lemmas the version stopped or started citing are replaced, every
        other list is shared."""
        node = _version(record, path, line, prover, run_id, self.root)
        lid = node["id"]
        lemmas = dict(self.lemmas)
        old = lemmas.get(lid)
        lemmas[lid] = node
        hashes = dict(self.hashes)
        tree = merkle.Merkle(lemmas, self.references, clean=True)
        cited = node["cited_lemmas"]
        if lid in cited or any(c in lemmas and hashes.get(c) is None for c in cited):
            hashes[lid] = None
        else:
            hashes[lid] = tree.node_hash(node, clean=True)
        if old is None:
            susp = None
            if self._suspension is not None:
                susp = suspension.with_lemma(
                    self._suspension, lid, node, hashes[lid],
                    [c for c in self.certificates
                     if str(c.get("lemma_id") or "").strip() == lid],
                    [r for _p, r in self.refutations
                     if str(r.get("lemma_id") or "").strip() == lid],
                    self.conflicts,
                )
            # Nothing cites a new lemma: its depth and ancestry are its
            # own, and no other lemma's moves.
            depth = ancestry = None
            if self._depth is not None:
                depth = dict(self._depth)
                depth[lid] = citations.depth_of(lemmas, depth, node)
            if self._ancestry is not None:
                ancestry = self._ancestry.with_lemma(lemmas, node)
            return self._replace(susp, self._carried_cited_by(lid, old, node),
                                 depth, ancestry, lemmas=lemmas, hashes=hashes)
        if old["statement_hash"] != node["statement_hash"]:
            # A reproof changes what lid cites, never who cites lid.
            for k in self.cited_by_map().get(lid, ()):
                if hashes.get(k) is not None:
                    hashes[k] = tree.node_hash(lemmas[k], clean=True)
        # A reproof citing what its old version cited moves no depth and
        # no ancestry; one citing something else moves those of everything
        # above it, so they are built again when next asked for.
        same = set(old["cited_lemmas"]) == set(node["cited_lemmas"])
        return self._replace(cited_by=self._carried_cited_by(lid, old, node),
                             depth=self._depth if same else None,
                             ancestry=self._ancestry if same else None,
                             lemmas=lemmas, hashes=hashes)

    def _carried_cited_by(self, lid: str, old: Optional[Dict[str, Any]],
                          node: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
        """This DAG's map of who cites whom with lid's version old replaced
        by node (old None for a new lemma); None when the map was never
        built, so the new DAG builds its own when asked."""
        if self._cited_by is None:
            return None
        before = set(old["cited_lemmas"]) if old is not None else set()
        after = set(node["cited_lemmas"])
        if before == after:
            return self._cited_by
        out = dict(self._cited_by)
        for c in before - after:
            kept = [x for x in out.get(c, ()) if x != lid]
            if kept:
                out[c] = kept
            else:
                out.pop(c, None)
        for c in after - before:
            out[c] = [*out.get(c, ()), lid]
        return out

    def with_certificate(self, line: Dict[str, Any]) -> "Dag":
        """This DAG with a certificate added. A commit certifies its lemma
        before writing it, and a certificate for a lemma the DAG does not
        hold yet changes no suspension, so the suspension is carried over;
        for a lemma it holds, it is recomputed when next asked for."""
        carried = self._suspension if line.get("lemma_id") not in self.lemmas else None
        return self._replace(carried, certificate_lines=[*self.certificates, line])

    def with_refutation(self, path: Path, record: Dict[str, Any]) -> "Dag":
        return self._replace(refutation_files=[*self.refutations, (path, record)])

    def without_refutations(self, paths: List[Path]) -> "Dag":
        gone = {str(p) for p in paths}
        return self._replace(refutation_files=[
            (p, r) for p, r in self.refutations if str(p) not in gone
        ])


# ----------------------------------------------------------------------------
# Writing: the run's streams
# ----------------------------------------------------------------------------
def _user_header(ctx) -> Dict[str, str]:
    """The header every stream file of this run starts with: whose it is
    (the config's user.id, user.name, user.email) and the run. Not the
    model: a resumed run may use another, so each record that depends on
    one names its own."""
    return {
        config.USER_ID_KEY: ctx.user,
        config.USER_NAME_KEY: config.get(config.USER_NAME_KEY) or "",
        config.USER_EMAIL_KEY: config.get(config.USER_EMAIL_KEY) or "",
        "run_id": ctx.run_id,
    }


def writer(ctx, kind: str) -> "streams.Writer":
    """The run's stream of one kind ("dags", "certificates",
    "computations"), made the first time it is asked for."""
    w = ctx.run.writers.get(kind)
    if w is None:
        w = streams.Writer(Path(ctx.paths.conjecture_root) / kind, ctx.user,
                           ctx.run_id, _user_header(ctx))
        ctx.run.writers[kind] = w
    return w


def new_lemma_id(ctx) -> str:
    """A new candidate's permanent id, <user_id>/<run_id>/<6 hex digits>:
    drawn again while the DAG holds it or the run has handed it out before
    (ctx.run.issued_ids, which a resumed run seeds from its checkpoint). No
    other run can draw it, since its run id is in it."""
    prefix = f"{ctx.user}/{ctx.run_id}/"
    with ctx.run.id_lock:
        while True:
            lemma_id = prefix + secrets.token_hex(3)
            if lemma_id not in ctx.run.issued_ids and lemma_id not in ctx.dag.lemmas:
                ctx.run.issued_ids.add(lemma_id)
                return lemma_id


def _record(ctx, kind: str, lemma_id: str, node: Dict[str, Any]) -> Dict[str, Any]:
    """A lemma (or reproof) record, in the order the stream writes it, its
    statement and proof normalized: the one place a lemma's text is
    normalized, so the load can hash it as stored. Its model is the one
    that wrote the proof (node["_model"], which the proof loop sets; this
    run's otherwise), left out when not known; no hash covers it."""
    model = str(node["_model"] if "_model" in node else _recorded_model(ctx))
    record = {
        "type": kind,
        "id": lemma_id,
        "statement": merkle.normalize(node["statement"]),
        "proof": merkle.normalize(node["proof"]),
        "cited_lemmas": [str(c) for c in node.get("cited_lemmas", [])],
        "cited_references": list(node.get("cited_references", [])),
        "code": [
            {"hash": e["hash"], "lang": e["lang"],
             "produced_output": e.get("produced_output")}
            for e in computations.entries(node)
        ],
        "proved_at": certificates.now_iso(),
    }
    if model:
        record["model"] = model
    return record


def _store_computations(ctx, node: Dict[str, Any]) -> None:
    """Copy the scripts a node cites into code/<user_id>/<run_id>/ and record
    their executions (computations.store): the first step of a commit, so a
    lemma is never in a stream without its scripts."""
    sources = node.get("_code_sources") or {}
    if computations.entries(node) and sources:
        computations.store(Path(ctx.paths.conjecture_root), ctx.user, ctx.run_id,
                           node, sources, writer(ctx, computations.DIR_NAME))


def _append_version(ctx, record: Dict[str, Any]) -> None:
    path = writer(ctx, DIR_NAME).append(record)
    with open(path, "rb") as f:
        line = f.read().count(b"\n")       # the record is the file's last line
    lid = str(record.get("id") or "")
    statement = str(record.get("statement") or "")
    old = ctx.dag.lemmas.get(lid)
    ctx.run.dag = ctx.dag.with_version(record, path, line, ctx.user, ctx.run_id)
    # Under the DAG lock, like the swap: the index sees the commits in the
    # order the DAG does. A reproof that keeps its statement changes no word.
    if ctx.run.search_index is not None and (old is None or old["statement"] != statement):
        ctx.run.search_index.put(lid, statement)


def commit_lemma_to_dag(
    ctx, lemma_id: str, node: Dict[str, Any], verified_hash: Optional[str],
    verbose: bool = True,
) -> str:
    """Certify node and commit it to the run's dags/ stream under the id
    its candidate was given (new_lemma_id); returns the id.

    The order is certificate, then lemma (_certify_before_write): a reader
    between the two sees a certificate for a lemma not in the DAG yet,
    which counts for nothing, never a lemma without its certificate. Under
    the DAG lock, so the threads of a parallel run commit one at a time."""
    with ctx.run.dag_lock:
        if lemma_id in ctx.dag.lemmas:
            # new_lemma_id never hands out an id the DAG holds, and a lemma
            # is committed once: this is a bug, not a state to recover from.
            raise ValueError(f"{lemma_id} is already in the DAG")
        _store_computations(ctx, node)
        _certify_before_write(ctx, lemma_id, node, verified_hash, verbose)
        _append_version(ctx, _record(ctx, LEMMA, lemma_id, node))
        return lemma_id


def commit_reproof(
    ctx, lemma_id: str, node: Dict[str, Any], verified_hash: Optional[str],
    verbose: bool = True,
) -> None:
    """Commit a re-proof of a lemma already in the DAG (proofs repair): a
    reproof record in the run's own stream, superseding the current
    version, certified first the way a new lemma is."""
    with ctx.run.dag_lock:
        current = ctx.dag.lemmas[lemma_id]
        _store_computations(ctx, node)
        _certify_before_write(ctx, lemma_id, node, verified_hash, verbose)
        record = _record(ctx, REPROOF, lemma_id, node)
        record["supersedes"] = version_of(current)
        _append_version(ctx, record)


def _certify_before_write(
    ctx, lemma_id: str, node: Dict[str, Any],
    verified_hash: Optional[str], verbose: bool = True,
) -> Optional[str]:
    """Record a certificate for a lemma this run just accepted, before the
    lemma is written: the first half of a commit. Returns the hash the
    lemma will have (None for a citation cycle).

    The hash is the one the lemma will have once it is written: the
    in-memory DAG with node in place under its id. verified_hash is the
    hash of the version the verifiers checked (_verified_hash); when the
    two differ — something it stands on changed between the verification
    and the commit — the certificate would cover a version nobody verified,
    so none is issued and the lemma stays suspended until proofs verify
    checks it. A certificate that cannot be written does not undo the
    acceptance: the lemma is committed anyway, suspended."""
    model = _recorded_model(ctx)
    try:
        h = ctx.dag.hash_of(lemma_id, node)
    except merkle.MerkleCycleError as e:
        ctx.ui.log(f"No certificate for {lemma_id}: {e}", verbose)
        return None
    if verified_hash is not None and h != verified_hash:
        ctx.ui.log(
            f"No certificate for {lemma_id}: something it stands on changed "
            f"between the verification and the commit, so the version "
            f"committed is not the one the verifiers checked. It stays "
            f"suspended until verified again: run proofs verify "
            f"{Path(ctx.paths.conjecture_root).name} {lemma_id}.",
            verbose,
        )
        return h
    try:
        line = certificates.record(writer(ctx, certificates.DIR_NAME),
                                   lemma_id, h, ctx.user, model)
    except OSError as e:
        ctx.ui.log(f"Could not write the certificate for {lemma_id}: {e}", verbose)
        return h
    ctx.run.dag = ctx.dag.with_certificate(line)
    how = f"model {model}" if model else "human acceptance"
    ctx.ui.log(f"Certificate for {lemma_id}: accepted by {ctx.user} ({how}).", verbose)
    return h


def _verified_hash(dag: "Dag", node: Dict[str, Any]) -> Optional[str]:
    """The Merkle hash of a proof as the verifiers checked it: the node's
    statement, proof and citations, over the DAG they were shown. The node
    is hashed under a key no lemma can have, so it neither replaces a
    lemma nor depends on the id it will be committed under. None when the
    hash cannot be computed (a citation cycle below it)."""
    try:
        return dag.hash_of("\x00verified", node)
    except merkle.MerkleCycleError:
        return None


# ----------------------------------------------------------------------------
# What the run tells the operator
# ----------------------------------------------------------------------------
def suspension_kinds(dag: "Dag") -> Dict[str, Set[str]]:
    """The suspended lemmas (see suspension.py) sorted by what brings each
    back, every lemma in exactly one kind, the first that holds of it:

      conflict     its versions state different things: a person deletes
                   all but one
      cycle        on a citation cycle, so it has no hash and no
                   certificate can be valid for it
      dangling     cites a lemma id no stream holds (pruned, or not
                   pulled) or a reference id references.md lacks: its hash
                   covers a statement that is not there
      refuted      a refutation counts against it: proofs repair
      uncertified  no valid certificate, and nothing else of its own:
                   proofs verify --uncertified
      cites        suspended only for citing a suspended lemma: its own
                   certificate stands, so it comes back with what it cites
    """
    susp = dag.suspension()
    kinds: Dict[str, Set[str]] = {
        k: set() for k in
        ("conflict", "cycle", "dangling", "refuted", "uncertified", "cites")
    }
    if not susp.suspended:
        return kinds
    certified = {
        str(c.get("lemma_id") or "") for c in dag.certificates
        if certificates.is_valid(c, dag.hashes.get(str(c.get("lemma_id") or "")))
    }
    ref_ids = {str(r.get("id")) for r in dag.references}
    for lid in susp.suspended:
        node = dag.lemmas[lid]
        if lid in dag.conflicts:
            kinds["conflict"].add(lid)
        elif dag.hashes.get(lid) is None:
            kinds["cycle"].add(lid)
        elif any(c not in dag.lemmas for c in node.get("cited_lemmas") or []) \
                or any(str(r) not in ref_ids
                       for r in node.get("cited_references") or []):
            kinds["dangling"].add(lid)
        elif lid in susp.refuted:
            kinds["refuted"].add(lid)
        elif lid not in certified:
            kinds["uncertified"].add(lid)
        else:
            kinds["cites"].add(lid)
    return kinds


def describe_suspension(dag: "Dag", name: str) -> str:
    """One line counting the suspended lemmas by kind (suspension_kinds),
    each with what brings it back; "" when none is suspended. Counts, never
    ids: a DAG can hold tens of thousands of suspended lemmas."""
    kinds = suspension_kinds(dag)
    total = sum(len(v) for v in kinds.values())
    if not total:
        return ""
    parts = [
        f"{len(kinds[kind]):,} {what}"
        for kind, what in (
            ("refuted", "refuted (proofs repair)"),
            ("uncertified", f"uncertified (proofs verify {name} --uncertified)"),
            ("cites", "only citing a suspended lemma"),
            ("dangling", "citing a lemma or reference that is not there "
                         "(pull, or proofs prune --with-dependents)"),
            ("conflict", "forked with different statements (delete all but "
                         "one version by hand)"),
            ("cycle", "on a citation cycle"),
        )
        if kinds[kind]
    ]
    return (f"{total:,} lemma(s) are suspended, hidden from the planner and "
            f"the prover: " + "; ".join(parts) + ".")


def _warn_suspended(
    ctx,
    susp: "suspension.Suspension",
    verbose: bool,
    shown: Optional[Dict[str, Any]] = None,
) -> None:
    """The run's warning for the suspended lemmas (see suspension.py): one
    line of counts by kind (describe_suspension), never a line per lemma.
    With shown (a dict the caller keeps for the run), told again only when
    the counts change."""
    if shown is not None and shown.get("susp") is susp:
        return
    line = describe_suspension(ctx.dag, Path(ctx.paths.conjecture_root).name)
    if shown is not None:
        shown["susp"] = susp
        if shown.get("line") == line:
            return
        shown["line"] = line
    if line:
        ctx.ui.log(line, verbose)


def _settled(ctx, dag: "Dag", suspended: Iterable[str], conjecture: str,
             verbose: bool) -> Optional[str]:
    """Whether the DAG settles the conjecture now (resolution.find): the
    kind, "proved" or "disproved", or None while it is open. A DAG that
    holds both a proof and a disproof settles nothing — it is inconsistent,
    and the run is told so and goes on."""
    found = resolution.find(dag.lemmas, set(suspended), conjecture)
    kinds = {r.kind for r in found}
    if len(kinds) > 1:
        ctx.ui.log(
            "The DAG both proves and disproves the conjecture ("
            + "; ".join(r.lemma_id for r in found)
            + "): at least one of these proofs is wrong. Treating the "
            "conjecture as open; proofs verify --full on each will show where.",
            verbose,
        )
        return None
    if not found:
        return None
    for r in found:
        ctx.ui.log(f"{resolution.describe(r)}.", verbose)
    return found[0].kind


def _committed_in_run(dag: "Dag", in_flight: Optional[Dict[str, Any]]) -> Optional[str]:
    """The in-flight lemma's id when the DAG already holds it, else None:
    how a resumed run tells that the lemma it held in flight was committed
    before it stopped."""
    lid = str((in_flight or {}).get("lemma_id") or "")
    return lid if lid in dag.lemmas else None


def _committed_uncertified(ctx, dag: "Dag", susp: "suspension.Suspension",
                           in_flight: Dict[str, Any]) -> Optional[str]:
    """The id of the lemma a checkpoint's in-flight lemma was committed as,
    when it was committed but never certified (the run stopped between the
    two): suspended, not refuted, and with no certificate anywhere. None
    otherwise — and then the resume goes ahead."""
    lid = _committed_in_run(dag, in_flight)
    if lid is None or lid not in susp.suspended or lid in susp.refuted:
        return None
    if any(str(c.get("lemma_id") or "") == lid for c in dag.certificates):
        return None
    return lid


def _uncertified_message(ctx, lemma_id: str) -> str:
    return (
        f"The lemma in flight was committed as {lemma_id} but never certified "
        f"(the last run stopped between the two). It is suspended until "
        f"certified: run proofs verify {Path(ctx.paths.conjecture_root).name} "
        f"{lemma_id}. Not re-proving it."
    )


# ----------------------------------------------------------------------------
# Walking the citations
# ----------------------------------------------------------------------------
def _citation_closure(dag: "Dag", lemma_id: str) -> List[str]:
    """The lemma's cited-lemma closure in dependency order: every lemma it
    cites, dependencies first, the target last — the order proofs verify
    --full verifies in. A cited id the DAG does not hold is skipped, and a
    citation cycle is cut at the first repeat."""
    return _post_order(dag.lemmas, [lemma_id], dag.lemmas.__contains__)


def dependents(dag: "Dag", start: Iterable[str]) -> Set[str]:
    """Every lemma that cites one of start, directly or transitively."""
    cited_by = dag.cited_by_map()
    out: Set[str] = set()
    queue = list(start)
    while queue:
        for dep in cited_by.get(queue.pop(), []):
            if dep not in out:
                out.add(dep)
                queue.append(dep)
    return out - set(start)


def _dependent_chain(dag: "Dag", start: str) -> List[str]:
    """start's transitive dependents in dependency order, each before the
    lemmas that cite it — the order proofs repair climbs in. start itself
    is not in the result; a citation cycle is cut at the first repeat."""
    in_set = dependents(dag, [start])
    return [lid for lid in _dependency_order(dag, in_set | {start}) if lid != start]


def _dependency_order(dag: "Dag", ids: Set[str]) -> List[str]:
    """The given lemmas in dependency order: every lemma after the lemmas
    it cites that are among them — the order proofs export writes in. A
    citation cycle is cut at the first repeat; sorted start points keep
    the result deterministic."""
    return _post_order(dag.lemmas, sorted(ids), ids.__contains__)


def _post_order(lemmas: Dict[str, Dict[str, Any]], roots: Iterable[str],
                keep) -> List[str]:
    """Depth first from each root in turn, through the cited lemmas keep
    admits, each lemma after everything it cites; a citation cycle is cut
    at the first repeat. Iterative: a citation chain can be far deeper
    than Python's recursion limit."""
    out: List[str] = []
    done: Set[str] = set()
    on_stack: Set[str] = set()
    for root in roots:
        if root in done or not keep(root):
            continue
        on_stack.add(root)
        work = [(root, iter(lemmas[root].get("cited_lemmas", [])))]
        while work:
            lid, cited = work[-1]
            for c in cited:
                if c not in done and c not in on_stack and keep(c):
                    on_stack.add(c)
                    work.append((c, iter(lemmas[c].get("cited_lemmas", []))))
                    break
            else:
                work.pop()
                on_stack.discard(lid)
                done.add(lid)
                out.append(lid)
    return out


def load_references(ctx) -> List[Dict[str, Any]]:
    """Load references.md — proofs parse's output (parsing.py).

    A strict JSON array of { id, slogan, "formal statement", reference } objects.
    The file is the conjecture's committed reference
    collection (see parsing.py): the maintainer parses it and
    commits it, every other user pulls it, and this module only ever reads
    it — a missing or corrupt file is never repaired here.
    A missing file returns [], and every call site degrades to prompts
    with no references section. A corrupt file warns and returns [] too —
    the right fix is for the maintainer to re-run proofs parse, not to
    hand the loop a subset. Ids (ref_N) are assigned by parsing.py,
    never here, so a prover that cites one is citing a name the file vouches
    for.
    """
    if not ctx.paths.references_file or not os.path.exists(ctx.paths.references_file):
        return []
    try:
        with open(ctx.paths.references_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text.startswith("```"):
            text = _FENCE_RE.sub("", text).strip()
        data = json.loads(text)
    except (OSError, json.JSONDecodeError) as e:
        ctx.ui.log(f"{ctx.paths.references_file} is not valid JSON ({e}); ignoring it. "
            f"Re-run proofs parse if you expected references here.")
        return []
    if not isinstance(data, list):
        ctx.ui.log(f"{ctx.paths.references_file} is not a JSON array; ignoring it.")
        return []
    return [
        r for r in data
        if isinstance(r, dict) and str(r.get("id") or "").strip()
    ]


