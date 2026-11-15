"""proofs migrate: a format-1 workspace's files, rewritten as streams.

Format 1 kept one file per user and kind:

    dags/<user_id>_dag.json        {"user.id", "user.name", "user.email",
                                    "lemmas": {lemma_id: node}}
    certificates/<user_id>.jsonl   an identity line, then one line per
                                    (lemma, hash, verifier, model), counted
    refutations/<user>--<lemma>--<verifier>--<time>.json   with "user_id"
    runs/<user_id>.jsonl           one line per execution outcome, counted
    code/<user_id>/<sha256>.<ext>  the scripts

A lemma was the pair (user_id, lemma_id), the user being the file it lived
in, and its hash was format 1's (_OldMerkle): the normalized statement and
proof themselves, the sorted statement hashes of what it cites, and its
computations when it had any.

Each user becomes one run per conjecture, with a run id derived from the
user id alone (run_id_for), so everyone who migrates the same files writes
the same files, and a migration interrupted part way and run again picks up
where it stopped. Two migrations of different commits write the same paths
with different contents: git reports the conflict, and the streams loader
stops on it, rather than the same lemmas arriving twice under two run ids.

  * dags: every lemma becomes a lemma record in its owner's run, under the
    id <user_id>/<run_id>/<old lemma_id>. Its statement and proof are
    copied untouched, so a proof's "by lemma_56 (Spark)" still reads onto
    Spark/<run_id>/lemma_56; its citations are rewritten to the new ids,
    resolved the way format 1 resolved them (pairs, and a bare id exactly
    one file owns).
  * certificates: one that was valid — its hash format 1's hash of the
    lemma — is issued again under the new hash, in its verifier's run, with
    its verifier, model, date and count. The new hash covers the same
    statement, proof and cited statements the verifier accepted. One that
    was not valid is dropped.
  * refutations: one that counted is written again with the new id and
    hash and the stream file of the version it rejects; one that did not
    is dropped.
  * runs/ becomes each user's computations stream (one record per old line,
    its count kept), and code/<user_id>/ moves to code/<user_id>/<run_id>/.
  * The old files, checkpoints, locks and temp files are deleted, the
    workspace marker goes to format 2, and the .gitignore block is
    rewritten. Nothing is committed: the person reviews and commits.

The new files are written before any old one is deleted, and the DAG files
— what the ids and the old hashes are computed from — are deleted last.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from ..workspace import config
from ..workspace import layout as workspace
from . import certificates, merkle, refutations, streams
from .dag import DIR_NAME as DAGS_DIRNAME
from .dag import LEMMA

OLD_DAG_SUFFIX = "_dag.json"
OLD_RUNS_DIRNAME = "runs"
CODE_DIRNAME = "code"
COMPUTATIONS_DIRNAME = "computations"
EXECUTION = "execution"

Pair = Tuple[str, str]

# A piece of a new id: format 1's lemma ids were model-written, and a new id
# is split on "/".
_ID_HOSTILE = re.compile(r"[^A-Za-z0-9._-]+")


def run_id_for(user_id: str) -> str:
    """The run a user's format-1 files become: 16 hex digits derived from
    the user id, the same for everyone who migrates."""
    return hashlib.sha256(f"proofs migrate\0{user_id}".encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Format 1's hash
# ----------------------------------------------------------------------------
class _OldMerkle:
    """Format 1's lemma hashes, keyed by (user_id, lemma_id): the
    canonical JSON of {"statement": normalized statement, "proof":
    normalized proof, "cited": the sorted, distinct statement hashes of what
    it cites[, "code": its computations, only when it has any]}."""

    def __init__(self, lemmas: Mapping[Pair, Mapping[str, Any]],
                 references: Iterable[Mapping[str, Any]]) -> None:
        self.lemmas = lemmas
        self.owners: Dict[str, List[Pair]] = {}
        for pair in lemmas:
            self.owners.setdefault(pair[1], []).append(pair)
        self.ref_hashes = {
            str(r.get("id") or "").strip(): merkle.reference_hash(r)
            for r in references if isinstance(r, Mapping) and str(r.get("id") or "").strip()
        }
        self.cache: Dict[Pair, str] = {}
        self.stack: List[Pair] = []

    def citations(self, node: Mapping[str, Any]) -> Tuple[List[Pair], List[str]]:
        """The lemma pairs and reference ids a node cites, as format 1 read
        them: a {"user_id", "lemma_id"} object (or a two-item list) is a
        lemma; a bare id in cited_lemmas or the legacy dependencies list
        that exactly one file owns is that lemma; any other bare id is a
        reference."""
        pairs: List[Pair] = []
        refs: List[str] = []
        for fld in ("cited_lemmas", "dependencies", "cited_references"):
            entries = node.get(fld)
            for entry in entries if isinstance(entries, list) else []:
                pair = _pair(entry)
                if pair is None and isinstance(entry, str) and entry.strip():
                    rid = entry.strip()
                    if fld != "cited_references" and len(self.owners.get(rid, [])) == 1:
                        pair = self.owners[rid][0]
                    elif rid not in refs:
                        refs.append(rid)
                if pair is not None and pair not in pairs:
                    pairs.append(pair)
        return pairs, refs

    def hash(self, pair: Pair) -> Optional[str]:
        """The pair's format-1 hash; None for a lemma of a citation cycle,
        which format 1 gave none."""
        try:
            return self._hash(pair)
        except merkle.MerkleCycleError:
            return None

    def _hash(self, pair: Pair) -> str:
        cached = self.cache.get(pair)
        if cached is not None:
            return cached
        if pair in self.stack:
            raise merkle.MerkleCycleError(pair)
        self.stack.append(pair)
        try:
            node = self.lemmas[pair]
            pairs, refs = self.citations(node)
            cited: List[str] = []
            for p in pairs:
                if p in self.lemmas:
                    self._hash(p)       # for the cycle check, as format 1 did
                    cited.append(merkle.statement_hash(self.lemmas[p].get("statement")))
                else:
                    cited.append(_old_unresolved("lemma", *p))
            for rid in refs:
                cited.append(self.ref_hashes.get(rid) or _old_unresolved("reference", rid))
            hashed: Dict[str, Any] = {
                "statement": merkle.normalize(node.get("statement")),
                "proof": merkle.normalize(node.get("proof")),
                "cited": sorted(set(cited)),
            }
            code = merkle.code_entries(node)
            if code:
                hashed["code"] = code
            digest = merkle.sha256_hex(merkle.canonical_json(hashed).encode("utf-8"))
        finally:
            self.stack.pop()
        self.cache[pair] = digest
        return digest


def _old_unresolved(kind: str, *parts: str) -> str:
    marker = {"unresolved_citation": {"kind": kind, "parts": list(parts)}}
    return merkle.sha256_hex(merkle.canonical_json(marker).encode("utf-8"))


def _pair(entry: Any) -> Optional[Pair]:
    if isinstance(entry, Mapping):
        uid, lid = entry.get("user_id"), entry.get("lemma_id")
    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
        uid, lid = entry
    else:
        return None
    if isinstance(uid, str) and isinstance(lid, str) and uid.strip() and lid.strip():
        return (uid.strip(), lid.strip())
    return None


# ----------------------------------------------------------------------------
# Reading format 1
# ----------------------------------------------------------------------------
def _read_json_lines(path: Path) -> List[Dict[str, Any]]:
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            raise streams.StreamError(path, n, f"not JSON ({e})")
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _identity(obj: Mapping[str, Any], user_id: str) -> Dict[str, str]:
    return {
        config.USER_ID_KEY: user_id,
        config.USER_NAME_KEY: str(obj.get(config.USER_NAME_KEY) or ""),
        config.USER_EMAIL_KEY: str(obj.get(config.USER_EMAIL_KEY) or ""),
    }


def _is_old_refutation(rec: Mapping[str, Any]) -> bool:
    return "user_id" in rec and "/" not in str(rec.get("lemma_id") or "")


def old_files(conjecture: Path) -> List[Path]:
    """The format-1 files a conjecture still holds (none once migrated):
    what makes a command refuse it (workspace.resolve) and what proofs
    migrate converts."""
    found: List[Path] = []
    found += sorted((conjecture / DAGS_DIRNAME).glob(f"*{OLD_DAG_SUFFIX}"))
    found += sorted(p for p in (conjecture / certificates.DIR_NAME).glob("*.jsonl") if p.is_file())
    found += sorted((conjecture / OLD_RUNS_DIRNAME).glob("*.jsonl"))
    code = conjecture / CODE_DIRNAME
    if code.is_dir():
        found += sorted(p for d in code.iterdir() if d.is_dir() for p in d.iterdir() if p.is_file())
    for p in sorted((conjecture / refutations.DIR_NAME).glob("*.json")):
        try:
            if _is_old_refutation(json.loads(p.read_text(encoding="utf-8"))):
                found.append(p)
        except (OSError, ValueError):
            continue
    return found


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------
def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_stream(base: Path, user_id: str, run_id: str, header: Mapping[str, Any],
                  records: List[Dict[str, Any]], root: Path) -> List[str]:
    """Write a run's whole stream of one kind — header first in every file,
    at most streams.MAX_FILE_BYTES a file, a record never split — over
    whatever an earlier attempt left there. Returns, per record, the file
    it is in (relative to root)."""
    directory = streams.run_dir(base, user_id, run_id)
    head = streams._dumps({"type": streams.HEADER, **header})
    files: List[List[str]] = [[]]
    size = len(head.encode("utf-8"))
    where: List[str] = []
    for rec in records:
        line = streams._dumps(rec)
        n = len(line.encode("utf-8"))
        if files[-1] and size + n > streams.MAX_FILE_BYTES:
            files.append([])
            size = len(head.encode("utf-8"))
        files[-1].append(line)
        size += n
        where.append(str((directory / streams.file_name(len(files))).relative_to(root)))
    for i, lines in enumerate(files, 1):
        _atomic_write(directory / streams.file_name(i), head + "".join(lines))
    for stale in streams.stream_files(directory)[len(files):]:
        stale.unlink()
    return where


def _new_id(pair: Pair, renamed: Mapping[Pair, str]) -> str:
    return f"{pair[0]}/{run_id_for(pair[0])}/{renamed.get(pair, pair[1])}"


# ----------------------------------------------------------------------------
# One conjecture
# ----------------------------------------------------------------------------
@dataclass
class Report:
    conjecture: Path
    lemmas: int = 0
    certificates: int = 0
    certificates_dropped: int = 0
    refutations: int = 0
    refutations_dropped: int = 0
    executions: int = 0
    scripts: int = 0
    notes: List[str] = field(default_factory=list)


def migrate_conjecture(conjecture: Path) -> Report:
    """Migrate one conjecture directory (see the module docstring)."""
    report = Report(conjecture)
    root = conjecture
    try:
        references = json.loads((root / workspace.REFERENCES_FILENAME).read_text(encoding="utf-8"))
        if not isinstance(references, list):
            references = []
    except (OSError, ValueError):
        references = []

    # --- the DAG files -----------------------------------------------------
    owners: Dict[str, Dict[str, Any]] = {}
    lemmas: Dict[Pair, Dict[str, Any]] = {}
    dag_files = sorted((root / DAGS_DIRNAME).glob(f"*{OLD_DAG_SUFFIX}"))
    for path in dag_files:
        user = path.name[: -len(OLD_DAG_SUFFIX)]
        data = json.loads(path.read_text(encoding="utf-8"))
        owners[user] = data
        for lid, node in (data.get("lemmas") or {}).items():
            if isinstance(node, dict):
                lemmas[(user, str(lid))] = node
    old = _OldMerkle(lemmas, references)

    # New ids end in the old id; one a "/" or a space would break is
    # cleaned, and the change reported (a proof naming it then does not
    # match it verbatim).
    renamed: Dict[Pair, str] = {}
    taken: Set[Pair] = set()
    for (user, lid) in lemmas:
        clean = _ID_HOSTILE.sub("_", lid).strip(".") or "_"
        base, n = clean, 2
        while (user, clean) in taken:
            clean, n = f"{base}-{n}", n + 1
        taken.add((user, clean))
        if clean != lid:
            renamed[(user, lid)] = clean
            report.notes.append(f"{user}:{lid} became {user}/…/{clean} (its id was not usable as is)")

    new_nodes: Dict[str, Dict[str, Any]] = {}
    by_owner: Dict[str, List[Dict[str, Any]]] = {}
    for pair, node in lemmas.items():
        pairs, refs = old.citations(node)
        rec = {
            "type": LEMMA,
            "id": _new_id(pair, renamed),
            "statement": merkle.normalize(node.get("statement")),
            "proof": merkle.normalize(node.get("proof")),
            "cited_lemmas": [_new_id(p, renamed) for p in pairs],
            "cited_references": refs,
            "code": merkle.code_entries(node),
            "proved_at": str(node.get("proved_at") or ""),
        }
        by_owner.setdefault(pair[0], []).append(rec)
        new_nodes[rec["id"]] = rec
        prover = node.get("last_prover_id")
        if prover and prover != pair[0]:
            report.notes.append(f"{rec['id']} was last proved by {prover}, not its owner")
    new_hashes = merkle.Merkle(new_nodes, references).all()
    report.lemmas = len(new_nodes)

    lemma_file: Dict[str, str] = {}
    for user, recs in by_owner.items():
        header = {**_identity(owners[user], user), "run_id": run_id_for(user)}
        where = _write_stream(root / DAGS_DIRNAME, user, run_id_for(user), header, recs, root)
        lemma_file.update({rec["id"]: f for rec, f in zip(recs, where)})

    def translate(pair: Pair, old_hash: Any) -> Optional[Tuple[str, str]]:
        """(new id, new hash) of a lemma a record names, when the record's
        hash was its format-1 hash; None otherwise."""
        if pair not in lemmas or not old_hash or old.hash(pair) != old_hash:
            return None
        new = _new_id(pair, renamed)
        h = new_hashes.get(new)
        return (new, h) if h else None

    # --- certificates ------------------------------------------------------
    cert_files = sorted(p for p in (root / certificates.DIR_NAME).glob("*.jsonl") if p.is_file())
    for path in cert_files:
        verifier = path.stem
        identity: Dict[str, Any] = {}
        recs = []
        for line in _read_json_lines(path):
            if "lemma_id" not in line:
                identity = identity or line
                continue
            t = translate((str(line.get("user_id") or ""), str(line.get("lemma_id") or "")),
                          line.get("hash"))
            if t is None:
                report.certificates_dropped += 1
                continue
            cert = {
                "type": certificates.TYPE, "lemma_id": t[0], "hash": t[1],
                "verifier": str(line.get("verifier") or verifier),
                "model": str(line.get("model") or ""),
                "date": str(line.get("date") or ""),
            }
            n = certificates.count(line)
            if n > 1:
                cert["count"] = n
            recs.append(cert)
        if recs:
            header = {**_identity(identity or owners.get(verifier, {}), verifier),
                      "run_id": run_id_for(verifier)}
            _write_stream(certificates.dir_for(root), verifier, run_id_for(verifier),
                          header, recs, root)
            report.certificates += len(recs)

    # --- refutations -------------------------------------------------------
    old_refutations: List[Path] = []
    names: Set[str] = set()
    for path in sorted(refutations.dir_for(root).glob("*.json")):
        try:
            ref = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(ref, dict) or not _is_old_refutation(ref):
            continue
        old_refutations.append(path)
        t = translate((str(ref.get("user_id") or ""), str(ref.get("lemma_id") or "")),
                      ref.get("hash"))
        if t is None:
            report.refutations_dropped += 1
            continue
        when = str(ref.get("date") or "")
        new = {
            "lemma_id": t[0], "hash": t[1],
            "verifier": str(ref.get("verifier") or ""),
            "model": str(ref.get("model") or ""), "date": when,
            "verdict": str(ref.get("verdict") or ""),
            "justification": str(ref.get("justification") or ""),
            "file": lemma_file.get(t[0], ""),
        }
        # Two rejections in one second keep two files, as refutations.record
        # names them; counted within this migration, so a re-run writes the
        # same names again.
        name = base = refutations.name_for(t[0], new["verifier"], when)
        n = 2
        while name in names:
            name, n = base[: -len(".json")] + f"-{n}.json", n + 1
        names.add(name)
        _atomic_write(refutations.dir_for(root) / name, refutations._dumps(new))
        report.refutations += 1

    # --- computations and scripts -------------------------------------------
    run_files = sorted((root / OLD_RUNS_DIRNAME).glob("*.jsonl"))
    for path in run_files:
        user = path.stem
        recs = []
        for line in _read_json_lines(path):
            rec = {
                "type": EXECUTION,
                "script_sha256": line.get("script_sha256"),
                "language": line.get("language"),
                "tool": line.get("tool"),
                "result": line.get("result"),
                "result_sha256": line.get("result_sha256"),
                "date": str(line.get("first_date") or ""),
                "wall_seconds": line.get("wall_seconds"),
                "platform": line.get("platform"),
            }
            n = certificates.count(line)
            if n > 1:
                rec["count"] = n
                rec["last_date"] = str(line.get("last_date") or "")
            recs.append(rec)
        if recs:
            header = {**_identity(owners.get(user, {}), user), "run_id": run_id_for(user)}
            _write_stream(root / COMPUTATIONS_DIRNAME, user, run_id_for(user), header, recs, root)
            report.executions += len(recs)
    code = root / CODE_DIRNAME
    if code.is_dir():
        for user_dir in sorted(d for d in code.iterdir() if d.is_dir()):
            for script in sorted(p for p in user_dir.iterdir() if p.is_file()):
                dest = user_dir / run_id_for(user_dir.name) / script.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(script, dest)
                report.scripts += 1

    # --- the old files, the DAG files last ----------------------------------
    for path in [*cert_files, *old_refutations, *run_files]:
        path.unlink()
    runs_dir = root / OLD_RUNS_DIRNAME
    if runs_dir.is_dir() and not any(runs_dir.iterdir()):
        runs_dir.rmdir()
    for directory in (root / DAGS_DIRNAME, certificates.dir_for(root), refutations.dir_for(root)):
        for pattern in ("*.checkpoint.json", "*.checkpoint.json.*.tmp", "*.lock", "*.tmp", ".*.tmp"):
            for junk in directory.glob(pattern):
                if junk.is_file():
                    junk.unlink()
    for path in dag_files:
        path.unlink()
    for legacy in [root / "dag.json", *sorted(root.glob("dag-*.json"))]:
        if legacy.is_file():
            report.notes.append(
                f"{legacy.name} is a DAG from before per-user files, which format 1 "
                f"no longer read either; left in place, not migrated")
    return report


# ----------------------------------------------------------------------------
# The workspace
# ----------------------------------------------------------------------------
def migrate_workspace(root: Path) -> List[Report]:
    """Migrate every conjecture of the workspace at root that holds format-1
    files, then mark the workspace format 2 and rewrite its .gitignore
    block."""
    reports = []
    conjectures = root / workspace.CONJECTURES_ROOT
    for conjecture in sorted(d for d in conjectures.iterdir() if d.is_dir()):
        if old_files(conjecture):
            reports.append(migrate_conjecture(conjecture))
    marker = root / workspace.MARKER_FILENAME
    marker.write_text(f"format = {workspace.WORKSPACE_FORMAT}\n", encoding="utf-8")
    gitignore = root / ".gitignore"
    text = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    new = workspace.update_gitignore(text)
    if new != text:
        gitignore.write_text(new, encoding="utf-8")
    return reports


def migrate_entry(*, dir: Optional[str], parser: Any) -> int:
    """proofs migrate [DIR]: DIR (default: the workspace the current
    directory is in, or the current directory) is the workspace root, the
    directory holding conjectures/."""
    if dir is not None:
        root = Path(dir).resolve()
    else:
        root = workspace.find_root() or Path.cwd().resolve()
    if not (root / workspace.CONJECTURES_ROOT).is_dir():
        parser.error(f"{root} has no {workspace.CONJECTURES_ROOT}/; name the workspace root")
    if workspace.marker_format(root) >= workspace.WORKSPACE_FORMAT and not any(
        old_files(d) for d in (root / workspace.CONJECTURES_ROOT).iterdir() if d.is_dir()
    ):
        print(f"{root} is already format {workspace.WORKSPACE_FORMAT}; nothing to migrate.")
        return 0
    try:
        reports = migrate_workspace(root)
    except (OSError, ValueError, streams.StreamError) as e:
        print(f"proofs migrate: stopped: {e}. Nothing is lost: run it again "
              f"once that is fixed; it picks up where it stopped.")
        return 1
    for r in reports:
        print(f"{r.conjecture.name}: {r.lemmas} lemma(s); {r.certificates} certificate "
              f"record(s) re-issued, {r.certificates_dropped} invalid dropped; "
              f"{r.refutations} refutation(s) re-issued, {r.refutations_dropped} "
              f"stale dropped; {r.executions} execution record(s); {r.scripts} script(s) moved.")
        for note in r.notes:
            print(f"  {note}")
    print(f"{root} is now format {workspace.WORKSPACE_FORMAT}. Review the changes "
          f"(git status) and commit them.")
    return 0
