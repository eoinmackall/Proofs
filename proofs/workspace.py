"""Per-conjecture run directories.

Layout:

    conjectures/
      algebra_example/
        conjecture.md              <- required, the only mandatory file
        dags/                      <- per-user DAG files: <user_id>_dag.json
        references/                <- maintainer-local inputs for parsing.py (gitignored)
        references.md              <- committed: strict-JSON refs, parsing.py's output
        comments.md                <- optional, gitignored: this user's notes, passed to the planner
        prover.md                  <- optional per-conjecture prompt override
      some_other_problem/
        conjecture.md

    agents/
      planner.md  selector.md  prover.md  verifier_1/2/3.md  reviser.md  <- project-wide defaults

Two decisions worth flagging, both changeable:

  * One DAG file per user, not per model or per run:
    conjectures/NAME/dags/<user_id>_dag.json. run_loop() merges every file in
    dags/ into the complete DAG at the top of every iteration, so a user
    started against an existing conjecture picks up wherever everyone else
    stopped, whatever wrote those lemmas: the proof is the artefact, and a
    lemma proved by one person and passed by the three verifiers is no less
    proved when another person arrives to continue. Each lemma stores the
    user_id and lemma_id that identify it, and a lemma_id is unique only
    within one user's file — which is why the merge is keyed by the pair
    (user_id, lemma_id). The run writes new lemmas to the current user's file
    only, and never to another user's.

    The cost is the same as the old per-model sharing: users running against
    the same conjecture are collaborating, not competing, and their results
    are entangled from the first shared lemma.

  * Prompt files are looked up in the conjecture directory first, then in
    agents/. Nothing changes unless you drop a file in; it just means a
    conjecture that needs, say, a prover primed for Brauer groups can have one
    without forking the defaults.

  * references/ (input) and references.md (output) have different owners.
    The input directory is the conjecture's maintainer's local working copy
    of the source material — gitignored, never committed. The output,
    references.md, is the committed reference collection: the maintainer
    writes it with parsing.py and commits it with the conjecture, and every
    other user pulls it and never regenerates it. The proving loop only ever
    reads references.md; a result a user needs that is missing is requested
    from the maintainer, who parses it in (.comments, "References").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:  # inside the proofs package (installed, or `python -m proofs`)
    from . import certificates
except ImportError:  # top-level modules (`python proofs/workspace.py`)
    import certificates

CONJECTURES_ROOT = "conjectures"
CONJECTURE_FILENAME = "conjecture.md"
# Per-user DAG files live in dags/, named <user_id>_dag.json (see the module
# docstring). dag.json at the conjecture root is the pre-per-user shared DAG:
# still on disk in old clones, read by nothing now.
DAGS_DIRNAME = "dags"
USER_DAG_SUFFIX = "_dag.json"
LEGACY_DAG_FILENAME = "dag.json"
# The maintainer's input: a directory of .tex/.md/plain-text files to parse
# (local, gitignored — the raw sources are never committed). Its output:
# references.md, a strict-JSON array of
# { id, slogan, "formal statement", reference } objects the maintainer
# commits with the conjecture; main.py feeds it to the planner, the prover
# and the verifiers, and only ever reads it.
REFERENCES_DIRNAME = "references"
REFERENCES_FILENAME = "references.md"
# Free-form operator notes on possible approaches to a proof or
# counterexample. Read once per run and handed to the planner verbatim on
# every iteration; the planner is the only agent that sees it, and a
# conjecture directory without the file runs exactly as before. Gitignored
# and never committed or shared: each user keeps their own copy in their
# own clone (see .comments, "Comments").
COMMENTS_FILENAME = "comments.md"
AGENTS_DIR = "agents"
# Every prompt the loop can load, and therefore every per-conjecture override
# that is meaningful. parsing.md is the standalone parser's (parsing.py's).
# The three verifier_*.md files are the three atomic verification steps
# (VERIFIER_AGENTS in main.py).
#
# Each entry maps the lookup name (what PROMPT_PATHS is keyed by, and what a
# per-conjecture override file is called) to the path the project-wide default
# lives at, relative to agents/. The parallel-mode agents keep their own
# subdirectory on disk but stay flat in the map, so an override is always just
# "drop a file next to conjecture.md" whatever the agent is.
PROMPT_FILES: Dict[str, str] = {
    "planner.md": "planner.md",
    "selector.md": "selector.md",
    "prover.md": "prover.md",
    "verifier_1.md": "verifier_1.md",
    "verifier_2.md": "verifier_2.md",
    "verifier_3.md": "verifier_3.md",
    "reviser.md": "reviser.md",
    "parsing.md": "parsing.md",
    "parallel_planner.md": "parallel/parallel_planner.md",
    "parallel_lemma_generator.md": "parallel/parallel_lemma_generator.md",
}


class ConjectureNotFound(Exception):
    """Raised with the list of what *does* exist, so the fix is obvious."""


@dataclass
class RunPaths:
    root: Path                  # conjectures/algebra_example
    conjecture: Path            # conjectures/algebra_example/conjecture.md
    dags_dir: Path              # conjectures/algebra_example/dags/ (every user's DAG file)
    dag: Path                   # conjectures/algebra_example/dags/<user_id>_dag.json (this run's)
    references_dir: Path        # conjectures/algebra_example/references/ (input)
    references: Path            # conjectures/algebra_example/references.md
    comments: Path              # conjectures/algebra_example/comments.md (optional, gitignored, per user)
    prompts: Dict[str, Path]    # "prover.md" -> resolved path
    overridden_prompts: List[str]


# A conjecture name proofs new accepts: a single directory name, the same
# shape as a user.id (config.py), so it never escapes conjectures/.
_CONJECTURE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ConjectureExists(Exception):
    """Raised by create() rather than touch a directory that is already there."""


def create(name: str, project_root: Path = Path(".")) -> Path:
    """Lay out a blank conjecture, conjectures/<name>/, for proofs new.

    The directories every tool expects (references/, dags/, certificates/)
    and the files: conjecture.md and comments.md empty, references.md the
    empty collection "[]" — load_references() reads the file as a strict
    JSON array, so a zero-byte file would warn on every run. An existing
    directory is refused, never filled in: it may hold someone's work.
    Returns the new conjecture directory.
    """
    if not _CONJECTURE_NAME.fullmatch(name):
        raise ValueError(
            f"{name!r} is not a conjecture name: start with a letter or "
            f"digit, then letters, digits, dots, underscores or hyphens."
        )
    root = project_root / CONJECTURES_ROOT / name
    if root.exists():
        raise ConjectureExists(f"{root} already exists.")
    root.mkdir(parents=True)
    for d in (REFERENCES_DIRNAME, DAGS_DIRNAME, certificates.DIR_NAME):
        (root / d).mkdir()
    (root / CONJECTURE_FILENAME).write_text("", encoding="utf-8")
    (root / COMMENTS_FILENAME).write_text("", encoding="utf-8")
    (root / REFERENCES_FILENAME).write_text("[]\n", encoding="utf-8")
    return root


def available(project_root: Path = Path(".")) -> List[str]:
    root = project_root / CONJECTURES_ROOT
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / CONJECTURE_FILENAME).is_file()
    )


def agents_dir_for(conjecture_root: Path, project_root: Path = Path(".")) -> Path:
    """The agents/ directory whose prompts a conjecture runs with.

    The nearest agents/ above the conjecture directory — the repository the
    conjecture lives in, conjectures/NAME being two levels below it — so a
    command run from anywhere finds the same prompts a command run from the
    repository root does. A conjecture kept outside any repository falls
    back to the agents/ beside this package (an editable install, or a
    checkout on PYTHONPATH), then to project_root's.
    """
    for d in [conjecture_root, *conjecture_root.parents]:
        if (d / AGENTS_DIR).is_dir():
            return d / AGENTS_DIR
    beside_package = Path(__file__).resolve().parent.parent / AGENTS_DIR
    if beside_package.is_dir():
        return beside_package
    return (project_root / AGENTS_DIR).resolve()


def resolve(
    name: str,
    user_id: str,
    project_root: Path = Path("."),
) -> RunPaths:
    """Turn a conjecture name and the run's user into every path the run needs.

    `name` is accepted in whichever form is convenient:
        algebra_example
        conjectures/algebra_example
        conjectures/algebra_example/          (trailing slash, from tab-completion)
        /abs/path/to/algebra_example

    A path is taken relative to the current directory, or as given when
    absolute; a bare name is looked up under ./conjectures/. The paths
    returned are absolute, so nothing downstream depends on the directory
    the command was run from.

    The run's DAG file is dags/<user_id>_dag.json in the conjecture directory:
    the user's share of the conjecture's proof. The run reads every file in
    dags/ (the complete DAG) and writes this file alone.

    The project-wide prompts are found from the conjecture, not from the
    current directory: agents/ in the nearest directory above the conjecture
    that has one (the repository the conjecture lives in), else beside this
    package, else under project_root — see agents_dir_for().
    """
    if not user_id:
        raise ValueError("resolve() needs a user_id to name the run's DAG file")
    candidate = Path(name.rstrip("/\\"))
    for guess in (candidate, project_root / CONJECTURES_ROOT / candidate.name):
        if guess.is_dir():
            root = guess.resolve()
            break
    else:
        raise ConjectureNotFound(
            f"No conjecture directory {name!r}. "
            f"Available: {', '.join(available(project_root)) or '(none)'}"
        )

    conjecture = root / CONJECTURE_FILENAME
    if not conjecture.is_file():
        raise ConjectureNotFound(
            f"{root} has no {CONJECTURE_FILENAME}. "
            f"Available: {', '.join(available(project_root)) or '(none)'}"
        )
    # load_file() returns "" for a missing file, which run_loop() reports as
    # "conjecture is empty" — true but misleading. Checking here means a typo
    # in the directory name says so.

    dags_dir = root / DAGS_DIRNAME
    dag = dags_dir / (user_id + USER_DAG_SUFFIX)

    agents = agents_dir_for(root, project_root)
    prompts: Dict[str, Path] = {}
    overridden: List[str] = []
    for name, rel in PROMPT_FILES.items():
        local = root / name
        if local.is_file():
            prompts[name] = local
            overridden.append(name)
        else:
            prompts[name] = agents / rel

    return RunPaths(root=root, conjecture=conjecture, dags_dir=dags_dir,
                    dag=dag,
                    references_dir=root / REFERENCES_DIRNAME,
                    references=root / REFERENCES_FILENAME,
                    comments=root / COMMENTS_FILENAME,
                    prompts=prompts, overridden_prompts=overridden)


def describe(paths: RunPaths) -> str:
    lines = [
        f"conjecture={paths.root}  "
        f"dag={paths.dag.relative_to(paths.root)}"
    ]
    if paths.dag.exists():
        lines.append(
            f"  ↻ resuming from existing {paths.dag.name} "
            f"(delete it for a clean run)"
        )
    if paths.dags_dir.is_dir():
        others = sorted(
            p.name for p in paths.dags_dir.iterdir()
            if p.is_file()
            and p.name.endswith(USER_DAG_SUFFIX)
            and p.name != paths.dag.name
        )
        if others:
            lines.append(
                f"  ⬡ other users' DAG files in {paths.dags_dir.name}/: "
                f"{', '.join(others)} — merged into the run, never written"
            )
    checkpoint = paths.dag.with_name(
        paths.dag.stem + ".checkpoint" + paths.dag.suffix
    )
    if checkpoint.exists():
        lines.append(
            f"  ♻ checkpoint {checkpoint.name} present — the run resumes "
            f"from it (--fresh to restart the budget)"
        )
    # Runs from before DAGs were per-user left dag.json and
    # dag-<backend>-<model>.json files here. They are not read any more, and
    # saying so beats letting someone conclude the lemmas were lost.
    legacy_shared = paths.root / LEGACY_DAG_FILENAME
    if legacy_shared.is_file():
        lines.append(
            f"  ⚠ ignoring {legacy_shared.name} — the pre-per-user shared "
            f"DAG; move it to {paths.dags_dir.name}/<user_id>{USER_DAG_SUFFIX} "
            f"to carry its lemmas into the run"
        )
    legacy = sorted(p.name for p in paths.root.glob("dag-*.json"))
    if legacy:
        lines.append(
            f"  ⚠ ignoring per-model DAG{'s' if len(legacy) > 1 else ''} "
            f"{', '.join(legacy)} — the per-user DAGs in {paths.dags_dir.name}/ "
            f"replaced them"
        )
    # Checkpoints now live beside the user's DAG file in dags/, so a
    # *.checkpoint.json at the conjecture root is an orphan of the shared
    # DAG: nothing reads it, and saying so beats it looking live.
    legacy_cp = sorted(p.name for p in paths.root.glob("*.checkpoint.json"))
    if legacy_cp:
        lines.append(
            f"  ⚠ ignoring stale checkpoint{'s' if len(legacy_cp) > 1 else ''} "
            f"{', '.join(legacy_cp)} — checkpoints now live beside the "
            f"user's DAG file in {paths.dags_dir.name}/"
        )
    if paths.references_dir.is_dir():
        n_files = sum(1 for p in paths.references_dir.iterdir() if p.is_file())
        lines.append(
            f"  references={paths.references_dir.name}/ ({n_files} file(s) for "
            f"parsing.py)"
        )
    if paths.references.is_file():
        try:
            entries = json.loads(paths.references.read_text(encoding="utf-8"))
            n = len(entries) if isinstance(entries, list) else -1
        except (OSError, json.JSONDecodeError):
            n = -1
        if n < 0:
            lines.append(
                f"  ⚠ references={paths.references.name} is not valid JSON; "
                f"the loop will ignore it"
            )
        else:
            lines.append(f"  references={paths.references.name} ({n} entries)")
    comments = paths.comments
    if comments.is_file() and comments.read_text(encoding="utf-8").strip():
        lines.append(
            f"  comments={comments.name} — yours alone (gitignored); "
            f"passed to the planner verbatim as suggestions"
        )
    if paths.overridden_prompts:
        lines.append(
            f"  ✎ local prompt overrides: {', '.join(paths.overridden_prompts)}"
        )
    return "\n".join(lines)
