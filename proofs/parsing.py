"""Standalone references parser.

Scans a conjecture's references/ directory for .tex/.md/plain-text files,
reads each one with the LLM (agents/parsing.md), extracts the theorem-level
results, and writes them to references.md as a strict JSON array:

    [
      {
        "id": "ref_1",
        "slogan": "one plain-English sentence",
        "formal statement": "the result, self-contained, in the file's notation",
        "reference": "Chapter 4, Theorem 4.3"
      }
    ]

Who maintains it
----------------
references.md is the conjecture's committed reference collection
(.comments, "References"). The conjecture's maintainer is the only one who
runs this tool: the maintainer keeps the sources in references/ (a local,
gitignored working copy — the raw files are never committed), parses them,
and commits the updated references.md with the conjecture. Every other user
pulls references.md and never regenerates it: a result they need that is
missing is requested from the maintainer, who parses it in. The proving loop
(main.py) only ever reads the file; nothing in it writes to it. In a clone
without the maintainer's references/ directory this tool is a clean no-op:
nothing to parse, and the pulled references.md is left untouched.

main.py then feeds that file to the loop: the prover sees the whole
collection and may cite any entry by id in cited_references, the verifier sees
the formal statements of exactly the cited entries, and the planner sees an
id + slogan shortlist while the collection stays below
main.PLANNER_REFERENCE_LIMIT.

This module is deliberately standalone: `python parsing.py --conjecture
<name>` runs before, or instead of, the proving loop. It reuses main.py's
LLM plumbing — reason/extract, the JSON-repair pipeline, the compaction
rescue — by importing it, so the two entry points share one definition of a
failed call instead of two.

Id stability across re-parses
-----------------------------
Ids (ref_N) are assigned here, never by the model, so a citation the prover
writes is a name this file vouches for. When references.md already exists, a
re-parsed result whose normalized formal statement matches an existing entry
keeps that entry's id: proved lemmas may cite ref_N in their
cited_references, and renumbering it
would dangle every proof that leans on it. Results newly derived this run
take the next free numbers. Entries in the previous file whose statements
are not re-derived this run are kept (union semantics): a flaky parse of one
file must not silently delete results other proofs cite. A re-parse is
therefore idempotent on a good run — same files in, same file out.

Usage
-----
    proofs parse conjectures/algebra_example
    proofs parse conjectures/algebra_example --model Qwen3.5-122B-Q4_K_M
    proofs parse conjectures/algebra_example --num-ctx 65536

(also: python proofs/parsing.py --conjecture algebra_example)

--conjecture names a directory under conjectures/ holding a conjecture.md,
as main.py does. Its references/ directory is the input; references.md,
beside it, is the output.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # inside the proofs package (installed, or `python -m proofs`)
    from . import config, llm_backend, main, workspace
except ImportError:  # top-level modules (`python proofs/parsing.py`)
    import config
    import llm_backend
    import main
    import workspace

# Files that could plausibly contain statements. Everything else in the
# directory (images, pdfs, .gitkeep) is skipped with a note, not an error:
# the directory is a drop zone, and a dropped-in figure is not a failure.
PARSABLE_SUFFIXES = (".tex", ".md", ".txt", ".latex", ".text", ".rst")

# A file larger than this fraction of the context cannot leave room for a
# reply, so it is skipped with a suggestion to split it rather than burning
# a call that is guaranteed to hit the ceiling.
MAX_FILE_FRACTION = 0.7

# Fallback schema for the extraction stage: the shape parsing.md asks for
# directly, with the id added afterwards by this tool.
_REFERENCE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slogan": {"type": "string"},
                    "formal statement": {"type": "string"},
                    "reference": {"type": "string"},
                },
                "required": ["slogan", "formal statement", "reference"],
            },
        },
    },
    "required": ["references"],
}


def _norm(statement: Any) -> str:
    """The identity of a result across re-parses.

    Whitespace collapsed and lowercased: a statement re-extracted from the
    same file comes back the same modulo line breaks and a nervous model's
    case changes, and that is the same result. Punctuation and wording that
    actually differs is treated as a different statement, which is the
    conservative reading — it costs a duplicate entry, never a renumber.
    """
    return " ".join(str(statement or "").split()).lower()


def _max_ref_id(references: List[Dict[str, Any]]) -> int:
    """Highest ref_N in a list of entries, 0 when there is none."""
    top = 0
    for r in references:
        m = re.fullmatch(r"ref_(\d+)", str(r.get("id") or ""))
        if m:
            top = max(top, int(m.group(1)))
    return top


def is_parsable(path: Path) -> bool:
    """A text file the model can read: a known text suffix, or no suffix at
    all (plain-text chapters often ship extension-less)."""
    if path.suffix.lower() in PARSABLE_SUFFIXES:
        return True
    return path.suffix == ""


def load_previous(path: Path, verbose: bool) -> List[Dict[str, Any]]:
    """The previous references.md, or [] when it is absent.

    This module owns the file, so a corrupt one is a harder stop than for
    main.load_references: starting fresh would renumber ids the DAG already
    cites. Absent is fine (first run); unparseable warns and starts fresh
    only after saying so, because a half-written file from an interrupted
    run is the usual cause and re-running on the remaining files is the fix.
    """
    if not path.is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                          flags=re.MULTILINE).strip()
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("top-level JSON is not an array")
        return [r for r in data if isinstance(r, dict)]
    except (OSError, ValueError, json.JSONDecodeError) as e:
        main.log(
            f"⚠️  existing {path.name} is not a JSON array of objects ({e}); "
            f"starting fresh. Ids the DAG may cite are gone — if that file "
            f"mattered, restore it and re-run.",
            verbose,
        )
        return []


def parse_file(
    system_prompt: str,
    text: str,
    filename: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """One file in, its theorem-level results out.

    Two stages, mirroring the loop's agents: parsing.md asks for strict JSON
    directly, so the reasoning reply is parsed as-is; the extraction stage
    only runs when that parse fails. An "id" field in the model's answer is
    never read — numbering is this module's job, and honouring a model-
    invented id could collide with an entry the previous run vouches for.
    """
    user = (
        "Parse this file and extract its theorem-level results.\n\n"
        f"[File: {filename}]\n{text}"
    )
    # reason() returns (content, status, partial); the partial work is
    # unused here — a file too long for one pass is skipped, not rescued.
    content, status, _ = main.reason(
        system_prompt, user, "parser", main.THINK["parser"], verbose
    )
    if status == "ceiling" and not content:
        main.log(
            f"  ⛔ {filename}: parsing hit the context wall: the file is too "
            f"long for one pass. Split it into chapters and re-run.",
            verbose,
        )
        return []
    if not content:
        main.log(f"  ⚠️  {filename}: the parser returned nothing.", verbose)
        return []

    res = main.parse_json_or_none(content)
    if res is None:
        res = main.extract(
            "Extract the references. Copy every result the text identifies "
            'into "references", preserving the order it presents them in, and '
            'copy each field (slogan, "formal statement", reference) '
            "verbatim where the text gives one. Do not add an id field.",
            content,
            _REFERENCE_SCHEMA,
            "parser",
            verbose,
        )
    res = res or {}
    items = res.get("references")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def make_entry(
    item: Dict[str, Any],
    rid: str,
    source: str,
    verbose: bool,
) -> Dict[str, Any]:
    """One model answer into one references.md row, in canonical order.

    The formal statement is the load-bearing field — it is what the prover
    reads as the result it is citing and what the verifier checks the proof
    against — so a result without one is dropped, not stored empty.
    """
    statement = str(item.get("formal statement") or "").strip()
    if not statement:
        main.log(
            f"  ⚠️  {source}: dropping a result with no formal statement "
            f"({str(item.get('slogan'))[:80]!r}).",
            verbose,
        )
        return {}
    return {
        "id": rid,
        "slogan": str(item.get("slogan") or "").strip(),
        "formal statement": statement,
        "reference": str(item.get("reference") or "").strip(),
    }


# Entry point, named run() because this module imports main and a function
# called main would shadow it: every main.reason / main.log call below would
# then resolve against the function instead of the module.
def run(argv: Optional[List[str]] = None, prog: Optional[str] = None) -> None:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Parse a conjecture's references/ directory into references.md.",
    )
    ap.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable console logging (default: --verbose)",
    )
    ap.add_argument(
        "--host", default=None,
        help="OpenAI-compatible base URL: a local llama-server, or a hosted "
             "API's base (e.g. https://api.openai.com, without the /v1 "
             "suffix — it is added automatically). Defaults to "
             "$LLAMA_HOST or localhost:8081 (8080 is taken by open-webui).",
    )
    ap.add_argument(
        "--api-key", default=None,
        help="API key sent as 'Authorization: Bearer <key>'. Needed for a "
             "hosted API; defaults to $LLM_API_KEY. Leave unset for a plain "
             "local llama-server (or one launched without --api-key).",
    )
    ap.add_argument(
        "--model", default=main.MODEL_NAME,
        help="Model name: the --alias llama-server was launched with, or the "
             "provider's model id (e.g. gpt-4o).",
    )
    ap.add_argument(
        "--backend", choices=("auto", "llamacpp", "openai"), default="auto",
        help=(
            "Which dialect of the OpenAI-compatible endpoint to speak. "
            "auto (default): probe /props — an answer means llama.cpp (full "
            "option set, thinking suppressed for schema calls), no answer "
            "means a generic endpoint (standard OpenAI fields). llamacpp / "
            "openai force one side of that decision, for when the probe "
            "can't see the real server."
        ),
    )
    ap.add_argument(
        "--conjecture", default=main.DEFAULT_CONJECTURE,
        help=f"Directory under {workspace.CONJECTURES_ROOT}/ containing "
             f"{workspace.CONJECTURE_FILENAME} (default: {main.DEFAULT_CONJECTURE})",
    )
    ap.add_argument(
        "--num-ctx", type=int, default=None,
        help=(
            "Context window in tokens. Defaults to the server's context — "
            "the -c llama-server was launched with, as probed (lower it if "
            "VRAM is tight) — or 65536 on a hosted --host API, where "
            "nothing is probed. An explicit value replaces that default; "
            "against a probed llama-server it is clamped to the ceiling."
        ),
    )
    args = ap.parse_args(argv)

    # The same backend setup as main.main(): probe first, then resolve the
    # context window — an explicit --num-ctx wins, clamped only against a
    # probed llama.cpp ceiling (a fact), or the server's own context when it
    # is not given.
    main.BACKEND = llm_backend.make_backend(
        args.model, args.host, main.REQUEST_TIMEOUT, api_key=args.api_key,
        backend=args.backend
    )
    main.PROFILE = main.BACKEND.probe()
    if main.PROFILE.auth_error:
        ap.error(main.PROFILE.auth_error)
    if args.num_ctx is None:
        num_ctx = main.PROFILE.context_limit
    elif main.PROFILE.probed_llama_cpp:
        num_ctx = min(args.num_ctx, main.PROFILE.context_limit)
    else:
        num_ctx = args.num_ctx
    main.NUM_CTX = num_ctx
    main.REASONING_OPTIONS["num_ctx"] = num_ctx
    main.EXTRACT_OPTIONS["num_ctx"] = num_ctx
    # The banner warns when an *explicit* --num-ctx is clamped; in the auto
    # case the resolution above has already made the two agree.
    main.log(llm_backend.describe(
        main.PROFILE, args.num_ctx if args.num_ctx is not None else num_ctx
    ), args.verbose)

    # The run's user, before the paths: resolve() names the run's DAG
    # file after it, and the description the operator sees mentions whose
    # file the run writes. Parsing itself only reads references/ and
    # writes references.md; the user is irrelevant to that, but the paths
    # are the same shape a run would use.
    try:
        user_id = config.ensure_user_id()
        paths = workspace.resolve(args.conjecture, user_id)
    except config.UserConfigError as e:
        ap.error(str(e))
        return
    except workspace.ConjectureNotFound as e:
        ap.error(str(e))
        return
    main.log(workspace.describe(paths), args.verbose)

    system_prompt = main.load_file(paths.prompts["parsing.md"])
    if not system_prompt:
        ap.error(
            f"prompt file {paths.prompts['parsing.md']} is missing or empty; "
            f"it cannot be an empty agent"
        )
        return

    files = (
        sorted(p for p in paths.references_dir.iterdir() if p.is_file())
        if paths.references_dir.is_dir()
        else []
    )
    parsable = [p for p in files if is_parsable(p)]
    for p in files:
        if not is_parsable(p):
            main.log(f"  ↷ skipping {p.name}: not a text file.", args.verbose)

    if not parsable:
        main.log(
            f"\nNothing to parse: {paths.references_dir} has no .tex/.md/"
            f"plain-text files.",
            args.verbose,
        )
        if paths.references.is_file():
            main.log(
                f"Leaving the existing {paths.references.name} untouched.",
                args.verbose,
            )
        return

    previous = load_previous(paths.references, args.verbose)
    by_stmt: Dict[str, Dict[str, Any]] = {}
    for r in previous:
        key = _norm(r.get("formal statement"))
        if key and key not in by_stmt:
            by_stmt[key] = r
    next_id = _max_ref_id(previous) + 1
    if previous:
        main.log(
            f"Previous {paths.references.name}: {len(previous)} entries "
            f"(ids continue at ref_{next_id}).",
            args.verbose,
        )

    entries: List[Dict[str, Any]] = []
    this_run: set = set()
    reused = 0
    for path in parsable:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            main.log(f"  ↷ {path.name}: empty, skipping.", args.verbose)
            continue
        est = main.PROFILE.estimate_tokens(text)
        if est > int(MAX_FILE_FRACTION * num_ctx):
            main.log(
                f"  ⚠️  skipping {path.name}: ~{est} tokens leaves no room for "
                f"a reply inside the {num_ctx} context. Split the file into "
                f"chapters and re-run.",
                args.verbose,
            )
            continue
        main.log(f"📖 Parsing {path.name} (~{est} tokens)…", args.verbose)
        try:
            items = parse_file(system_prompt, text, path.name, args.verbose)
        except llm_backend.ContextLengthError as e:
            # A file this long for the window is a mis-sized --num-ctx, not
            # a bad file: say so once, with the server's own number, and
            # stop. references.md keeps its previous contents — re-run
            # with the corrected budget and everything re-parses.
            main.log(main._context_length_message(e, main.NUM_CTX))
            raise SystemExit(1)
        for item in items:
            statement = str(item.get("formal statement") or "").strip()
            key = _norm(statement)
            if not key:
                continue  # make_entry reports it when it lands
            if key in this_run:
                main.log(
                    f"  ↷ {path.name}: duplicate statement already extracted "
                    f"this run; keeping the first.",
                    args.verbose,
                )
                continue
            this_run.add(key)
            old = by_stmt.get(key)
            if old is not None:
                rid = str(old["id"])
                reused += 1
                # The reused id keeps its formal statement verbatim. _norm
                # matched the two up to case and whitespace, but the Merkle
                # hash of a cited reference is case-sensitive and keeps line
                # breaks (merkle.reference_hash), so the fresh extraction's
                # text would move the hash and void every certificate on a
                # lemma that cites it. The slogan and the source are not in
                # the hash, and are refreshed from the new extraction.
                item = {
                    **item,
                    "formal statement": str(old.get("formal statement") or ""),
                }
            else:
                rid = f"ref_{next_id}"
                next_id += 1
            entry = make_entry(item, rid, path.name, args.verbose)
            if entry:
                entries.append(entry)
        main.log(f"   ↳ {len(items)} result(s) extracted from {path.name}.",
                 args.verbose)

    # Union semantics: a re-parse must not delete what a previous run
    # established and the DAG may cite. Statements not re-derived this run
    # (their file was deleted, or the model missed them) stay in the file.
    kept = [
        {k: v for k, v in r.items() if k != "tags"}
        for r in previous
        if _norm(r.get("formal statement"))
        and _norm(r.get("formal statement")) not in this_run
    ]
    for r in kept:
        main.log(
            f"  ↻ keeping {r.get('id')} from the previous file: not "
            f"re-derived this run ({str(r.get('reference'))[:60]!r}).",
            args.verbose,
        )

    # Canonical order: numerical id order. Ids are reused or continue from
    # the previous file, so sorting keeps the file stable across re-parses —
    # an entry moves only when it is first derived, and a good re-parse
    # leaves the file byte-identical even if the model's within-file order
    # drifts.
    def _id_num(r: Dict[str, Any]) -> int:
        m = re.fullmatch(r"ref_(\d+)", str(r.get("id") or ""))
        return int(m.group(1)) if m else -1

    final = sorted(entries + kept, key=_id_num)
    previous_json = json.dumps(
        [r for r in previous if isinstance(r, dict)], indent=2
    )
    final_json = json.dumps(final, indent=2)
    if final_json != previous_json:
        with open(paths.references, "w", encoding="utf-8") as f:
            f.write(final_json + "\n")
        main.log(
            f"\n✅ Wrote {paths.references}: {len(final)} entries "
            f"({len(entries)} derived this run, {len(kept)} kept, "
            f"{reused} ids reused, {len(entries) - reused} new). It is the "
            f"conjecture's committed collection: commit it (and push) so the "
            f"other users' runs see the new entries — they pull this file "
            f"rather than regenerating it.",
            args.verbose,
        )
    else:
        main.log(
            f"\n✅ {paths.references} unchanged: {len(final)} entries "
            f"({len(entries)} re-derived, {len(kept)} kept, "
            f"{reused} ids reused).",
            args.verbose,
        )


if __name__ == "__main__":
    run()
