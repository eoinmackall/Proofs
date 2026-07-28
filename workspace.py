"""Per-conjecture run directories.

Layout:

    conjectures/
      curve_indices/
        conjecture.md              <- required, the only mandatory file
        dag-ollama-qwen3.6_27b.json
        dag-ollama-gemma4_31b.json
        dag-llamacpp-qwen3.5_122b.json
        prover.md                  <- optional per-conjecture prompt override
      some_other_problem/
        conjecture.md

    planner.md  prover.md  verifier_a.md  verifier_b.md   <- project-root defaults

Two decisions worth flagging, both changeable:

  * The DAG filename carries the backend and model. You are running the same
    conjecture against four models; run_loop() reloads the DAG at the top of
    every iteration, so a single shared dag.json would let each model resume
    from the previous one's lemmas and quietly invalidate the comparison.
    Pass --dag explicitly to override.

  * Prompt files are looked up in the conjecture directory first, then the
    project root. Nothing changes unless you drop a file in; it just means a
    conjecture that needs, say, a prover primed for Brauer groups can have one
    without forking the defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

CONJECTURES_ROOT = "conjectures"
CONJECTURE_FILENAME = "conjecture.md"
PROMPT_FILES = ("planner.md", "prover.md", "verifier_a.md", "verifier_b.md")


class ConjectureNotFound(Exception):
    """Raised with the list of what *does* exist, so the fix is obvious."""


@dataclass
class RunPaths:
    root: Path                  # conjectures/curve_indices
    conjecture: Path            # conjectures/curve_indices/conjecture.md
    dag: Path                   # conjectures/curve_indices/dag-<backend>-<model>.json
    prompts: Dict[str, Path]    # "prover.md" -> resolved path
    overridden_prompts: List[str]


def slugify(text: str) -> str:
    """Model tags contain characters that have no business in a filename.

    qwen3.6:27b            -> qwen3.6_27b
    hf.co/user/Model:Q4_K_M -> hf.co_user_Model_Q4_K_M
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


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
    backend: str,
    model: str,
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

    if dag_override:
        dag = Path(dag_override)
    else:
        dag = root / f"dag-{slugify(backend)}-{slugify(model)}.json"

    prompts: Dict[str, Path] = {}
    overridden: List[str] = []
    for fname in PROMPT_FILES:
        local = root / fname
        if local.is_file():
            prompts[fname] = local
            overridden.append(fname)
        else:
            prompts[fname] = project_root / fname

    return RunPaths(root=root, conjecture=conjecture, dag=dag,
                    prompts=prompts, overridden_prompts=overridden)


def describe(paths: RunPaths) -> str:
    lines = [f"conjecture={paths.root}  dag={paths.dag.name}"]
    if paths.dag.exists():
        lines.append(
            f"  ↻ resuming from existing {paths.dag.name} "
            f"(delete it for a clean run)"
        )
    if paths.overridden_prompts:
        lines.append(
            f"  ✎ local prompt overrides: {', '.join(paths.overridden_prompts)}"
        )
    return "\n".join(lines)
