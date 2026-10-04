"""Per-conjecture run directories.

Layout:

    conjectures/
      algebra_example/
        conjecture.md              <- required, the only mandatory file
        dag.json                   <- shared by every model and backend
        references/                <- optional: .tex/.md files for parsing.py
        references.md              <- optional: strict-JSON refs, parsing.py's output
        comments.md                <- optional: operator notes, passed to the planner as suggestions
        prover.md                  <- optional per-conjecture prompt override
      some_other_problem/
        conjecture.md

    agents/
      planner.md  selector.md  prover.md  verifier_1/2/3.md  reviser.md  <- project-wide defaults

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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

CONJECTURES_ROOT = "conjectures"
CONJECTURE_FILENAME = "conjecture.md"
DAG_FILENAME = "dag.json"
# parsing.py's input: a directory of .tex/.md/plain-text files to parse.
# Its output: references.md, a strict-JSON array of
# { id, slogan, "formal statement", reference, tags } objects that main.py
# feeds to the planner, the prover and the verifiers.
REFERENCES_DIRNAME = "references"
REFERENCES_FILENAME = "references.md"
# Free-form operator notes on possible approaches to a proof or
# counterexample. Read once per run and handed to the planner verbatim on
# every iteration; the planner is the only agent that sees it, and a
# conjecture directory without the file runs exactly as before.
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
    dag: Path                   # conjectures/algebra_example/dag.json
    references_dir: Path        # conjectures/algebra_example/references/ (input)
    references: Path            # conjectures/algebra_example/references.md
    comments: Path              # conjectures/algebra_example/comments.md (optional)
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
        algebra_example
        conjectures/algebra_example
        conjectures/algebra_example/          (trailing slash, from tab-completion)
        /abs/path/to/algebra_example
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
    for name, rel in PROMPT_FILES.items():
        local = root / name
        if local.is_file():
            prompts[name] = local
            overridden.append(name)
        else:
            prompts[name] = project_root / AGENTS_DIR / rel

    return RunPaths(root=root, conjecture=conjecture, dag=dag,
                    references_dir=root / REFERENCES_DIRNAME,
                    references=root / REFERENCES_FILENAME,
                    comments=root / COMMENTS_FILENAME,
                    prompts=prompts, overridden_prompts=overridden)


def describe(paths: RunPaths) -> str:
    lines = [f"conjecture={paths.root}  dag={paths.dag.name}"]
    if paths.dag.exists():
        lines.append(
            f"  ↻ resuming from existing {paths.dag.name} "
            f"(delete it for a clean run)"
        )
    checkpoint = paths.dag.with_name(
        paths.dag.stem + ".checkpoint" + paths.dag.suffix
    )
    if checkpoint.exists():
        lines.append(
            f"  ♻ checkpoint {checkpoint.name} present — the run resumes "
            f"from it (--fresh to restart the budget)"
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
            f"  comments={comments.name} — passed to the planner verbatim "
            f"as suggestions"
        )
    if paths.overridden_prompts:
        lines.append(
            f"  ✎ local prompt overrides: {', '.join(paths.overridden_prompts)}"
        )
    return "\n".join(lines)
