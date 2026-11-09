"""Computations a lemma's proof rests on: the scripts, their results, runs/.

A proof may rest a step on a script the prover ran (run_code, execution.py).
The proof itself stays prose — it says what the script checks and why that
suffices — and the lemma carries the computations it cites in one more
stored field:

    "code": [
      {"hash": "<sha256 of the script>", "lang": "sage",
       "produced_output": <the script's RESULT, as JSON>}
    ]

produced_output is the harness's record of the RESULT line, never the
model's: the prover names the runs it relies on by run_id, and
from_runs() reads each one's run.json from the user's scratch directory.
The field is written only when it is non-empty, and the Merkle hash takes
it in only then (merkle.py), so a lemma with no computation hashes exactly
as it did before computations existed.

When the lemma is committed, its scripts are copied out of scratch into the
conjecture, content-addressed, and each run that produced one is recorded:

    conjectures/NAME/code/<user_id>/<sha256>.sage | .py     committed
    conjectures/NAME/runs/<user_id>.jsonl                   committed

A script is stored under the user whose run produced it, and named for its
hash, so two users never write the same file and the same script is stored
once. A line of runs/<user_id>.jsonl is one execution outcome, keyed by
(script, tool name, tool version, tool path, RESULT): the same outcome
again bumps its count, so a file stays small however often a script is run,
and a later re-execution (proofs verify --run-code) that matches adds to a
count rather than a line. The key carries the tool's path, the Nix store
path when the tool came from the flake, so two builds of one Sage version
are told apart.

The verifiers read every cited script and its produced_output (context()),
but run nothing.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:  # inside the proofs package
    from . import codetools, locking
except ImportError:  # top-level modules
    import codetools
    import locking

CODE_DIRNAME = "code"
RUNS_DIRNAME = "runs"
RUNS_FIELDS = ("script_sha256", "language", "tool", "result", "result_sha256",
               "first_date", "last_date", "count", "wall_seconds", "platform")


def _extension(lang: str) -> str:
    for spec in codetools.TOOLS.values():
        if spec.language == lang:
            return spec.extension
    raise ValueError(f"no code tool for language {lang!r}")


def short(entry_or_hash: Any) -> str:
    """The 8-hex id a proof refers to a script by: [code:<id>]."""
    h = entry_or_hash.get("hash") if isinstance(entry_or_hash, Mapping) else entry_or_hash
    return str(h or "")[:8]


def entries(node: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The lemma's computations, as well-formed entries; [] for none."""
    raw = node.get("code")
    if not isinstance(raw, list):
        return []
    return [e for e in raw if isinstance(e, dict) and e.get("hash") and e.get("lang")]


def result_hash(result: Any) -> str:
    return hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# ----------------------------------------------------------------------------
# From the prover's run ids to entries
# ----------------------------------------------------------------------------

def from_runs(
    scratch_dir: Path, run_ids: Iterable[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Path], List[str]]:
    """(entries, sources, problems) for the runs a prover cites.

    entries are the node's code field; sources maps each script hash to the
    run directory it came from (what store() copies, and what context()
    reads before the commit); problems are what is wrong, as feedback the
    prover can act on — a run id that names no run of this user, a run
    that failed, two runs of one script that disagree."""
    out: List[Dict[str, Any]] = []
    sources: Dict[str, Path] = {}
    problems: List[str] = []
    for raw in run_ids:
        run_id = str(raw or "").strip()
        if not run_id or "/" in run_id or run_id.startswith("."):
            problems.append(f"{raw!r} is not a run_id")
            continue
        run_dir = scratch_dir / run_id
        try:
            log = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            problems.append(f"{run_id} names no run of yours (no such run_id)")
            continue
        if log.get("status") != "ok":
            problems.append(
                f"{run_id} did not succeed ({log.get('failure') or 'failed'}); "
                f"cite only runs whose status is ok"
            )
            continue
        sha = str(log.get("script_sha256") or "")
        entry = {"hash": sha, "lang": str(log.get("language") or ""),
                 "produced_output": log.get("result")}
        if sha in sources:
            prior = next(e for e in out if e["hash"] == sha)
            if result_hash(prior["produced_output"]) != result_hash(entry["produced_output"]):
                problems.append(
                    f"two runs of script {short(sha)} gave different "
                    f"RESULTs; make the script deterministic (sort its "
                    f"output, seed its randomness) and cite one run"
                )
            continue
        sources[sha] = run_dir
        out.append(entry)
    out.sort(key=lambda e: e["hash"])
    return out, sources, problems


_REFERENCE_RE = re.compile(r"\[code:([^\]\s]*)\]")


def unknown_references(proof: str, code: Iterable[Mapping[str, Any]]) -> List[str]:
    """What is wrong with the proof's [code:<id>] references: each one must
    name a script the proof cites (by the start of its hash, at least 8
    hex digits' worth of it as run_code reports). [] when they all do."""
    ids = {str(e["hash"]) for e in code}
    bad = []
    for ref in _REFERENCE_RE.findall(proof or ""):
        if len(ref) < 8 or not any(h.startswith(ref.lower()) for h in ids):
            bad.append(f"the proof refers to [code:{ref}], which is none of "
                       f"the scripts it cites")
    return bad


# ----------------------------------------------------------------------------
# Scripts on disk
# ----------------------------------------------------------------------------

def script_path(conjecture_dir: Path, entry: Mapping[str, Any]) -> Optional[Path]:
    """Where the committed script of entry is (any user's code/ directory),
    or None when no copy is on disk."""
    name = str(entry["hash"]) + _extension(str(entry["lang"]))
    hits = sorted(glob.glob(str(conjecture_dir / CODE_DIRNAME / "*" / glob.escape(name))))
    return Path(hits[0]) if hits else None


def _source_script(run_dir: Path, lang: str) -> Path:
    return run_dir / ("script" + _extension(lang))


def load_script(
    conjecture_dir: Path, entry: Mapping[str, Any],
    sources: Optional[Mapping[str, Path]] = None,
) -> Optional[str]:
    """The script's text, if the bytes on disk still hash to entry's hash:
    from the run directory when sources has it (before the commit), else
    from the committed copy. None when there is no intact copy."""
    candidates: List[Path] = []
    if sources and entry["hash"] in sources:
        candidates.append(_source_script(sources[entry["hash"]], str(entry["lang"])))
    committed = script_path(conjecture_dir, entry)
    if committed is not None:
        candidates.append(committed)
    for p in candidates:
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(data).hexdigest() == entry["hash"]:
            return data.decode("utf-8", errors="replace")
    return None


def _tool_label(conjecture_dir: Path, entry: Mapping[str, Any],
                sources: Optional[Mapping[str, Path]]) -> str:
    """"sage 10.4" for the tool a run of the script used: from its run log
    before the commit, else from any runs/ line for it."""
    if sources and entry["hash"] in sources:
        try:
            log = json.loads((sources[entry["hash"]] / "run.json").read_text(encoding="utf-8"))
            tool = log.get("tool") or {}
            return f"{tool.get('name', '')} {tool.get('version', '')}".strip()
        except (OSError, ValueError):
            pass
    for line in load_runs(conjecture_dir):
        if line.get("script_sha256") == entry["hash"]:
            tool = line.get("tool") or {}
            return f"{tool.get('name', '')} {tool.get('version', '')}".strip()
    return ""


def context(
    conjecture_dir: Path, node: Mapping[str, Any],
    sources: Optional[Mapping[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """The computations as the verifiers are shown them: each script's id,
    language, the tool that ran it, the script itself and its
    produced_output. A script with no intact copy on disk is shown as
    missing — which a verifier must treat as an unchecked step."""
    shown = []
    for e in entries(node):
        script = load_script(conjecture_dir, e, sources)
        shown.append({
            "id": short(e),
            "language": e["lang"],
            "ran_with": _tool_label(conjecture_dir, e, sources) or "unknown",
            "script": script if script is not None else
                      "(MISSING: no copy of this script on disk matches its hash)",
            "produced_output": e.get("produced_output"),
        })
    return shown


def store(
    conjecture_dir: Path, user_id: str, node: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> None:
    """Copy the node's scripts out of scratch into code/<user_id>/, and
    record the runs that produced them in runs/<user_id>.jsonl. A script
    already committed (by anyone) is not copied again. Called before the
    lemma is written, so the lemma is never in a DAG file without its
    scripts."""
    for e in entries(node):
        run_dir = sources.get(e["hash"])
        if run_dir is None:
            continue
        if script_path(conjecture_dir, e) is None:
            dest_dir = conjecture_dir / CODE_DIRNAME / user_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / (e["hash"] + _extension(e["lang"]))
            data = _source_script(run_dir, e["lang"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != e["hash"]:
                raise ValueError(f"{run_dir}: the script no longer matches its hash")
            tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        try:
            log = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        record_run(conjecture_dir, user_id, log)


# ----------------------------------------------------------------------------
# runs/<user_id>.jsonl
# ----------------------------------------------------------------------------

def runs_file(conjecture_dir: Path, user_id: str) -> Path:
    return conjecture_dir / RUNS_DIRNAME / f"{user_id}.jsonl"


def _read_lines(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def load_runs(conjecture_dir: Path) -> List[Dict[str, Any]]:
    """Every user's runs/ lines, together."""
    out: List[Dict[str, Any]] = []
    for p in sorted((conjecture_dir / RUNS_DIRNAME).glob("*.jsonl")):
        out.extend(_read_lines(p))
    return out


def _key(line: Mapping[str, Any]) -> Tuple[Any, ...]:
    tool = line.get("tool") or {}
    return (line.get("script_sha256"), tool.get("name"), tool.get("version"),
            tool.get("path"), line.get("result_sha256"))


def record_run(conjecture_dir: Path, user_id: str, log: Mapping[str, Any]) -> None:
    """Record one execution (a run.json) in runs/<user_id>.jsonl: a new line
    for a new outcome, a count bumped for one already there."""
    path = runs_file(conjecture_dir, user_id)
    tool = log.get("tool") or {}
    date = str(log.get("started_at") or "")
    line = {
        "script_sha256": log.get("script_sha256"),
        "language": log.get("language"),
        "tool": {"name": tool.get("name"), "version": tool.get("version"),
                 "path": tool.get("path")},
        "result": log.get("result"),
        "result_sha256": result_hash(log.get("result")),
        "first_date": date,
        "last_date": date,
        "count": 1,
        "wall_seconds": log.get("wall_seconds"),
        "platform": log.get("platform"),
    }
    with locking.file_lock(path):
        lines = _read_lines(path)
        for old in lines:
            if _key(old) == _key(line):
                old["count"] = int(old.get("count") or 0) + 1
                old["last_date"] = max(str(old.get("last_date") or ""), date)
                break
        else:
            lines.append(line)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(
            "".join(json.dumps({k: l.get(k) for k in RUNS_FIELDS},
                               ensure_ascii=False) + "\n" for l in lines),
            encoding="utf-8",
        )
        os.replace(tmp, path)


# ----------------------------------------------------------------------------
# proofs status and proofs prune
# ----------------------------------------------------------------------------

def executions(conjecture_dir: Path, script_hash: str,
               lines: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """The runs/ lines of one script, every user's."""
    lines = load_runs(conjecture_dir) if lines is None else lines
    return [l for l in lines if l.get("script_sha256") == script_hash]


def describe(conjecture_dir: Path, node: Mapping[str, Any],
             lines: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """One line per computation of node, for proofs status: its id, its
    recorded output, and how often, with what, it has been executed —
    and how many of those executions reproduced the recorded output."""
    lines = load_runs(conjecture_dir) if lines is None else lines
    out = []
    for e in entries(node):
        want = result_hash(e.get("produced_output"))
        runs = executions(conjecture_dir, e["hash"], lines)
        same = sum(int(l.get("count") or 0) for l in runs if l.get("result_sha256") == want)
        other = sum(int(l.get("count") or 0) for l in runs if l.get("result_sha256") != want)
        tools = sorted({f"{(l.get('tool') or {}).get('name')} {(l.get('tool') or {}).get('version')}"
                        for l in runs})
        present = "" if script_path(conjecture_dir, e) else "  SCRIPT MISSING"
        out.append(
            f"[code:{short(e)}] {e['lang']}  RESULT "
            f"{json.dumps(e.get('produced_output'), sort_keys=True)[:80]}  "
            f"executions: {same} reproduced"
            + (f", {other} differed" if other else "")
            + (f" ({', '.join(tools)})" if tools else "")
            + present
        )
    return out


# Scratch runs younger than this are kept by proofs prune: a run in
# progress may be about to cite one.
SCRATCH_KEEP_SECONDS = 7 * 24 * 3600


def prune(
    conjecture_dir: Path, user_id: str, cited: Iterable[str],
    scratch_dir: Optional[Path] = None, now: Optional[float] = None,
) -> Tuple[List[Path], int, List[Path]]:
    """Drop what no lemma rests on any more, from the user's own files only:
    scripts in code/<user_id>/ whose hash no lemma cites, runs/ lines for
    them, and scratch runs older than SCRATCH_KEEP_SECONDS (a cited
    script was copied out of scratch when its lemma was committed).
    Returns (scripts removed, runs/ lines removed, scratch runs removed).
    The caller holds every DAG file's lock, so no commit is half done."""
    import shutil
    import time as _time
    cited = set(cited)
    removed: List[Path] = []
    code_dir = conjecture_dir / CODE_DIRNAME / user_id
    if code_dir.is_dir():
        for p in sorted(code_dir.iterdir()):
            if p.is_file() and p.suffix in {s.extension for s in codetools.TOOLS.values()} \
                    and p.stem not in cited:
                p.unlink()
                removed.append(p)
    dropped_lines = 0
    path = runs_file(conjecture_dir, user_id)
    if path.is_file():
        with locking.file_lock(path):
            lines = _read_lines(path)
            kept = [l for l in lines if l.get("script_sha256") in cited]
            dropped_lines = len(lines) - len(kept)
            if dropped_lines:
                tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                tmp.write_text(
                    "".join(json.dumps({k: l.get(k) for k in RUNS_FIELDS},
                                       ensure_ascii=False) + "\n" for l in kept),
                    encoding="utf-8",
                )
                os.replace(tmp, path)
    old_runs: List[Path] = []
    if scratch_dir is not None and scratch_dir.is_dir():
        now = _time.time() if now is None else now
        for d in sorted(scratch_dir.iterdir()):
            try:
                age = now - d.stat().st_mtime
            except OSError:
                continue
            if d.is_dir() and age > SCRATCH_KEEP_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
                old_runs.append(d)
    return removed, dropped_lines, old_runs
