"""A role's scripts: the files run_code writes and edits for one agent.

Each proof loop has a directory of its own under the run's scratch, and in
it one directory per agent that runs code:

    conjectures/NAME/scratch/<user_id>/<run_id>/<loop dir>/
        prover/                    kept until the lemma settles
        reviser/                   removed when its call ends
        verifier_1/ … coding_verifier/   removed when its call ends
            s1.sage                the script, as it last ran
            s1.run.json            its last run's log (execution.Runner)
            s1.stdout.txt, s1.stderr.txt

The harness names the files (s1, s2, …): run_code with code writes a new
one, and run_code with base and edits rewrites one in place — or, with
new_file, writes the edited copy under a new name. A file is only worth
keeping while its text is in a context that can edit it, so a reviser's
and a verifier's files go when the call that wrote them ends, and at each
prover attempt the prover's directory keeps only the files the last proof
cited (the carry-over, which shows them to the prover again). The loop
directory goes when the lemma settles; its name is saved with the loop's
in-flight state, so a resumed run finds it again.

A prover cites a script by its script id (the start of its SHA-256, as
run_code reports it): resolve() finds the file whose current text has that
hash and whose last run of that text was ok.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from . import codetools

# The roles whose files outlive the call that wrote them.
KEPT_ROLES = ("prover",)

_NAME_RE = re.compile(r"^s(\d+)$")
# A run_id from before script ids (<stamp>-<sha8>-<rand>): read as its sha8.
_RUN_ID_RE = re.compile(r"^\d{8}t\d{6}z-([0-9a-f]{8})-[0-9a-f]{4}$")
_ID_RE = re.compile(r"^[0-9a-f]{8,64}$")


def new_loop_dir_name() -> str:
    return secrets.token_hex(4)


def loop_dir(ctx: Any) -> Path:
    """The calling loop's code directory: the proof loop's own (set by
    run_proof_loop), else one per thread, for a verifier that runs outside
    a proof loop (proofs verify)."""
    if ctx.loop.code_dir is not None:
        return ctx.loop.code_dir
    return ctx.code.runner.scratch_dir / ctx.loop.agent


def _language(path: Path) -> str:
    for spec in codetools.TOOLS.values():
        if path.suffix == spec.extension:
            return spec.language
    return ""


def log_path(script: Path) -> Path:
    return script.with_name(script.stem + ".run.json")


def read_log(script: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(log_path(script).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


class Scripts:
    """One role's directory of scripts."""

    def __init__(self, path: Path):
        self.path = path

    def files(self) -> Dict[str, Path]:
        """name -> script, for every script in the directory."""
        out: Dict[str, Path] = {}
        if not self.path.is_dir():
            return out
        for p in self.path.iterdir():
            if _NAME_RE.match(p.stem) and _language(p) and p.is_file():
                out[p.stem] = p
        return out

    def next_name(self) -> str:
        """A name no file here has: one past the highest."""
        numbers = [int(_NAME_RE.match(n).group(1)) for n in self.files()]
        return f"s{max(numbers, default=0) + 1}"

    def edit(self, base: str, edits: Any) -> Tuple[str, str]:
        """(language, text): base's script with edits applied, each
        {"old", "new"} matched against the file as it stands. Raises
        ValueError, with a message for the model, when base names no file
        or an edit does not apply."""
        script = self.files().get(str(base).strip())
        if script is None:
            names = ", ".join(sorted(self.files())) or "none yet"
            raise ValueError(f"there is no file {base!r}; your files: {names}")
        if not isinstance(edits, list) or not edits:
            raise ValueError('pass "edits", a list of {"old", "new"} replacements')
        text = script.read_text(encoding="utf-8")
        spans: List[Tuple[int, int, str]] = []
        for i, e in enumerate(edits, 1):
            old = e.get("old") if isinstance(e, dict) else None
            new = e.get("new", "") if isinstance(e, dict) else None
            if not isinstance(old, str) or not old or not isinstance(new, str):
                raise ValueError(f'edit {i}: pass "old" (non-empty) and "new", both strings')
            count = text.count(old)
            if count != 1:
                raise ValueError(
                    f"edit {i}: its old text appears {count} times in {base}; "
                    f"it must appear exactly once (quote more of the line)"
                    if count else
                    f"edit {i}: its old text does not appear in {base}"
                )
            start = text.index(old)
            spans.append((start, start + len(old), new))
        spans.sort()
        for (_s1, e1, _n1), (s2, _e2, _n2) in zip(spans, spans[1:]):
            if s2 < e1:
                raise ValueError("two edits overlap; merge them into one")
        for start, end, new in reversed(spans):
            text = text[:start] + new + text[end:]
        return _language(script), text

    def resolve(self, ids: Iterable[Any]) -> Tuple[
            List[Dict[str, Any]], Dict[str, Path], List[str], List[str]]:
        """(entries, sources, problems, names) for the script ids a proof
        cites: the node's code field, each script's file by its hash (what
        the verifiers read and the commit copies), what is wrong as
        feedback the prover can act on, and the names of the files cited."""
        current: List[Tuple[str, Path, str]] = [
            (name, p, sha256(p)) for name, p in sorted(self.files().items())
        ]
        entries: List[Dict[str, Any]] = []
        sources: Dict[str, Path] = {}
        problems: List[str] = []
        names: List[str] = []
        for raw in ids:
            sid = str(raw or "").strip().lower()
            if sid.startswith("[code:") and sid.endswith("]"):
                sid = sid[6:-1]
            m = _RUN_ID_RE.match(sid)
            if m:
                sid = m.group(1)
            if not _ID_RE.match(sid):
                problems.append(f"{raw!r} is not a script id (the 8 hex "
                                f"digits run_code reports)")
                continue
            matches = [(n, p, h) for n, p, h in current if h.startswith(sid)]
            if not matches:
                problems.append(
                    f"{sid} is not the current text of any of your files (a "
                    f"file you edited since has a new id): run it and cite "
                    f"the id that run reports")
                continue
            ok = []
            for name, p, h in matches:
                log = read_log(p)
                if log and log.get("script_sha256") == h and log.get("status") == "ok":
                    ok.append((name, p, h, log))
            if not ok:
                name = matches[0][0]
                log = read_log(matches[0][1]) or {}
                why = ((log.get("failure") or "failed").split(" — ")[0]
                       if log.get("script_sha256") == matches[0][2]
                       else "it has not been run since it was last edited")
                problems.append(f"{sid} ({name}) did not succeed ({why}); "
                                f"cite only scripts whose last run is ok")
                continue
            results = {json.dumps(log.get("result"), sort_keys=True) for *_x, log in ok}
            if len(results) > 1:
                problems.append(
                    f"two runs of script {sid} gave different RESULTs; make "
                    f"the script deterministic (sort its output, seed its "
                    f"randomness) and cite one")
                continue
            name, p, h, log = ok[0]
            if name not in names:
                names.append(name)
            if h in sources:
                continue
            sources[h] = p
            entries.append({"hash": h, "lang": _language(p),
                            "produced_output": log.get("result")})
        entries.sort(key=lambda e: e["hash"])
        return entries, sources, problems, names

    def keep_only(self, names: Iterable[str]) -> None:
        """Delete every file but the named scripts and their run files."""
        keep = set(names)
        if not self.path.is_dir():
            return
        for p in self.path.iterdir():
            if p.name.split(".", 1)[0] in keep:
                continue
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)

    def remove(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def listing(self) -> str:
        """Every file's current text, for a request that re-sends the task
        with the tool calls written in (a fold, a compaction rescue): the
        calls show the edits, this shows where they led. "" for none."""
        parts = []
        for name, p in sorted(self.files().items(),
                              key=lambda kv: int(kv[0][1:])):
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            parts.append(f"{name} ({_language(p)}, script id "
                         f"{sha256(p)[:8]}):\n```{_language(p)}\n"
                         f"{text.rstrip()}\n```")
        if not parts:
            return ""
        return "Your files as they stand now:\n\n" + "\n\n".join(parts)


def carried_block(entries: List[Mapping[str, Any]], names: List[str],
                  sources: Mapping[str, Path]) -> str:
    """The carry-over: the scripts the last proof cited, as the next prover
    attempt reads them. "" for none."""
    if not entries:
        return ""
    by_hash = {e["hash"]: e for e in entries}
    lines = ["Computations your last proof cited:"]
    seen = set()
    for name in names:
        h = next((h for h, p in sources.items() if p.stem == name), None)
        if h is None or h in seen:
            continue
        seen.add(h)
        e = by_hash[h]
        try:
            text = sources[h].read_text(encoding="utf-8")
        except OSError:
            continue
        lines.append(
            f"- {name}  [code:{h[:8]}]  {e['lang']}  RESULT: "
            f"{json.dumps(e.get('produced_output'), sort_keys=True)}\n"
            f"```{e['lang']}\n{text.rstrip()}\n```"
        )
    if len(lines) == 1:
        return ""
    lines.append("To reuse one unchanged, cite its id again in code_runs.\n"
                 "To fix one, call run_code with base=<name> and edits.")
    return "\n".join(lines)
