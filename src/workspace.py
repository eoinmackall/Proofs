"""Per-conjecture run directories.

Layout:

    conjectures/
      curve_indices/
        conjecture.md              <- required, the only mandatory file
        dag.json                   <- shared by every model and backend
        prover.md                  <- optional per-conjecture prompt override
      some_other_problem/
        conjecture.md

    agents/
      planner.md  prover.md  verifier.md  reviser.md      <- project-wide defaults

Two decisions worth flagging, both changeable:

  * One DAG per conjecture, not per model. run_loop() reloads it at the top of
    every iteration, so a model started against an existing dag.json picks up
    wherever the last one stopped, whatever wrote those lemmas: the proof is
    the artefact, and a lemma proved by a 27B model and passed by both
    verifiers is no less proved when a 122B model arrives to continue.

    The cost is that this is no longer a controlled comparison. Four models
    run against the same directory are collaborating, not competing, and their
    results are entangled from the first shared lemma. If you want them
    isolated again — to ask which model gets furthest alone — give each its
    own file with --dag; the tagged name this used to generate was
    dag-<backend>-<model>.json.

  * Prompt files are looked up in the conjecture directory first, then in
    agents/. Nothing changes unless you drop a file in; it just means a
    conjecture that needs, say, a prover primed for Brauer groups can have one
    without forking the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

CONJECTURES_ROOT = "conjectures"
CONJECTURE_FILENAME = "conjecture.md"
DAG_FILENAME = "dag.json"
AGENTS_DIR = "agents"
PROMPT_FILES = ("planner.md", "prover.md", "verifier.md", "reviser.md")


class ConjectureNotFound(Exception):
    """Raised with the list of what *does* exist, so the fix is obvious."""


@dataclass
class RunPaths:
    root: Path                  # conjectures/curve_indices
    conjecture: Path            # conjectures/curve_indices/conjecture.md
    dag: Path                   # conjectures/curve_indices/dag.json
    prompts: Dict[str, Path]    # "prover.md" -> resolved path
    overridden_prompts: List[str]


def available(project_root: Path = Path(".")) -> List[str]:
    root = project_root / CONJECTURES_ROOT
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / CONJECTURE_FILENAME).is_file()
    )


def resolve(
    name: str,
    dag_override: Optional[str] = None,
    project_root: Path = Path("."),
) -> RunPaths:
    """Turn a conjecture name into every path the run needs.

    `name` is accepted in whichever form is convenient:
        curve_indices
        conjectures/curve_indices
        conjectures/curve_indices/          (trailing slash, from tab-completion)
        /abs/path/to/curve_indices
    """
    candidate = Path(name.rstrip("/\\"))
    for guess in (candidate, project_root / CONJECTURES_ROOT / candidate.name):
        if guess.is_dir():
            root = guess
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

    dag = Path(dag_override) if dag_override else root / DAG_FILENAME

    prompts: Dict[str, Path] = {}
    overridden: List[str] = []
    for fname in PROMPT_FILES:
        local = root / fname
        if local.is_file():
            prompts[fname] = local
            overridden.append(fname)
        else:
            prompts[fname] = project_root / AGENTS_DIR / fname

    return RunPaths(root=root, conjecture=conjecture, dag=dag,
                    prompts=prompts, overridden_prompts=overridden)


def describe(paths: RunPaths) -> str:
    lines = [f"conjecture={paths.root}  dag={paths.dag.name}"]
    if paths.dag.exists():
        lines.append(
            f"  ↻ resuming from existing {paths.dag.name} "
            f"(delete it for a clean run)"
        )
    # Runs from before the DAG was shared left dag-<backend>-<model>.json
    # files here. They are not read any more, and saying so beats letting
    # someone conclude the lemmas were lost.
    legacy = sorted(
        p.name for p in paths.root.glob("dag-*.json") if p != paths.dag
    )
    if legacy:
        lines.append(
            f"  ⚠ ignoring per-model DAG{'s' if len(legacy) > 1 else ''} "
            f"{', '.join(legacy)} — rename one to {DAG_FILENAME} to carry it "
            f"forward, or pass --dag"
        )
    if paths.overridden_prompts:
        lines.append(
            f"  ✎ local prompt overrides: {', '.join(paths.overridden_prompts)}"
        )
    return "\n".join(lines)
