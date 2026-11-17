"""Computations a lemma's proof rests on: the scripts, their results, and
the record of their executions.

A proof may rest a step on a script the prover ran (run_code, execution.py).
The proof itself stays prose — it says what the script checks and why that
suffices — and the lemma record carries the computations it cites:

    "code": [
      {"hash": "<sha256 of the script>", "lang": "sage",
       "produced_output": <the script's RESULT, as JSON>}
    ]

produced_output is the harness's record of the RESULT line, never the
model's: the prover names the scripts it relies on by their script id, and
files.Scripts.resolve() reads each one's last run log from the prover's
directory in scratch. The Merkle hash takes the field in (merkle.py).

When the lemma is committed, its scripts are copied out of scratch into the
conjecture, content-addressed, under the user and the run that ran them, and
the execution that produced each one is recorded in the run's computations
stream (core/streams.py):

    conjectures/NAME/code/<user_id>/<run_id>/<sha256>.sage | .py      committed
    conjectures/NAME/computations/<user_id>/<run_id>/000001.jsonl     committed

so who ran a script, in which run and under which model (the stream's
header), is always on record. An execution record is

    {"type": "execution", "script_sha256", "language",
     "tool": {"name", "version", "path"}, "result", "result_sha256",
     "date", "wall_seconds", "platform"}

one per execution, never changed: a later re-execution (proofs verify
--run-code) that matches is a record of its own, and how often a script was
run, and with what, is counted when it is asked for (describe()). A record
proofs migrate converted from format 1, where one line counted every
matching execution, keeps that "count" (and its "last_date"). The tool's
path, the Nix store path when the tool came from the flake, tells two builds
of one Sage version apart.

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
from typing import Any, Dict, Iterable, List, Mapping, Optional

from . import codetools, files
from ...core import certificates, streams

CODE_DIRNAME = "code"
DIR_NAME = "computations"
EXECUTION = "execution"


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
    """Where the committed script of entry is (any run's code/ directory),
    or None when no copy is on disk."""
    name = str(entry["hash"]) + _extension(str(entry["lang"]))
    hits = sorted(glob.glob(str(conjecture_dir / CODE_DIRNAME / "*" / "*" / glob.escape(name))))
    return Path(hits[0]) if hits else None


def load_script(
    conjecture_dir: Path, entry: Mapping[str, Any],
    sources: Optional[Mapping[str, Path]] = None,
) -> Optional[str]:
    """The script's text, if the bytes on disk still hash to entry's hash:
    from its file in scratch when sources has it (before the commit), else
    from the committed copy. None when there is no intact copy."""
    candidates: List[Path] = []
    if sources and entry["hash"] in sources:
        candidates.append(sources[entry["hash"]])
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
    before the commit, else from any execution record of it."""
    if sources and entry["hash"] in sources:
        log = files.read_log(sources[entry["hash"]])
        if log is not None:
            tool = log.get("tool") or {}
            return f"{tool.get('name', '')} {tool.get('version', '')}".strip()
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
    conjecture_dir: Path, user_id: str, run_id: str, node: Mapping[str, Any],
    sources: Mapping[str, Path], writer: "streams.Writer",
) -> None:
    """Copy the node's scripts out of scratch into code/<user_id>/<run_id>/,
    and record the executions that produced them in the run's computations
    stream (writer). A script already committed (by anyone) is not copied
    again. Called before the lemma is written, so the lemma is never in a
    stream without its scripts."""
    for e in entries(node):
        source = sources.get(e["hash"])
        if source is None:
            continue
        if script_path(conjecture_dir, e) is None:
            dest_dir = conjecture_dir / CODE_DIRNAME / user_id / run_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / (e["hash"] + _extension(e["lang"]))
            tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
            tmp.write_bytes(source.read_bytes())
            os.replace(tmp, dest)
        log = files.read_log(source)
        if log is not None:
            record_run(writer, log)


# ----------------------------------------------------------------------------
# The computations streams
# ----------------------------------------------------------------------------

def load_runs(conjecture_dir: Path) -> List[Dict[str, Any]]:
    """Every execution record of every run."""
    return [
        rec for _u, _r, _p, _n, rec in streams.records(conjecture_dir / DIR_NAME)
        if rec.get("type") == EXECUTION
    ]


def record_run(writer: "streams.Writer", log: Mapping[str, Any]) -> None:
    """Record one execution (a run.json) in the run's computations stream."""
    tool = log.get("tool") or {}
    writer.append({
        "type": EXECUTION,
        "script_sha256": log.get("script_sha256"),
        "language": log.get("language"),
        "tool": {"name": tool.get("name"), "version": tool.get("version"),
                 "path": tool.get("path")},
        "result": log.get("result"),
        "result_sha256": result_hash(log.get("result")),
        "date": str(log.get("started_at") or ""),
        "wall_seconds": log.get("wall_seconds"),
        "platform": log.get("platform"),
    })


# ----------------------------------------------------------------------------
# proofs status
# ----------------------------------------------------------------------------

def executions(conjecture_dir: Path, script_hash: str,
               lines: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """The execution records of one script, every run's."""
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
        same = sum(certificates.count(l) for l in runs if l.get("result_sha256") == want)
        other = sum(certificates.count(l) for l in runs if l.get("result_sha256") != want)
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
