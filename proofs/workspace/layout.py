"""The proofs workspace and its per-conjecture run directories.

A workspace is a directory holding a .proofs marker file, laid out by
proofs init (init()); every command finds it by walking up from the current
directory to the first one with the marker (find_root()), the way git finds
.git. Conjecture paths below are relative to that root.

Layout:

    .proofs                        <- the marker: "format = 2"
    .proofs-local.json             <- gitignored: this machine's code limits (limits.py) and run
                                      settings (settings.py), written by init
    .gitignore                     <- holds the proofs block (GITIGNORE_RULES)
    conjectures/
      algebra_example/
        conjecture.md              <- required, the only mandatory file
        dags/<user_id>/<run_id>/000001.jsonl          <- committed: each run's lemma records (dag.py)
        certificates/<user_id>/<run_id>/000001.jsonl  <- committed: each run's acceptances (certificates.py)
        computations/<user_id>/<run_id>/000001.jsonl  <- committed: each run's code executions (computations.py)
        checkpoints/<user_id>/<run_id>/  <- gitignored: each run's checkpoint and lock (checkpoint.py)
        refutations/               <- committed: one file per rejection, created on first write
        references/                <- maintainer-local inputs for proofs parse (gitignored)
        references.md              <- committed: strict-JSON refs, proofs parse's output
        comments.md                <- optional, gitignored: this user's notes, passed to the planner
        prover.md                  <- optional per-conjecture prompt override
        flake.nix, flake.lock      <- optional, committed: the maintainer's code tools
        tools.lock                 <- committed with them: the tools runs may use (codetools.py)
        limits.json                <- optional, gitignored: this machine's limits here (limits.py)
        scratch/<user_id>/<run_id>/  <- gitignored: every code execution's script and log, per run
        code/<user_id>/<run_id>/   <- committed: the scripts cited proofs rest on (computations.py)
      some_other_problem/
        conjecture.md

    proofs/agents/                 <- inside the package, installed with it
      planner.md  prover.md  verifier_1/2/3.md  reviser.md  <- default prompts

Three decisions worth flagging, all changeable:

  * One stream per run, append-only (core/streams.py): no two runs, on one
    machine or many, ever write the same file, so git merges are additions.
    Every command reads every run's streams into one complete DAG
    (core/dag.py), so a user who pulls an existing conjecture picks up
    wherever everyone else stopped, whatever wrote those lemmas: the proof
    is the artefact, and a lemma one person proved and certified is no less
    proved when another person arrives to continue. A lemma's id —
    <user_id>/<run_id>/<hex> — is unique everywhere by construction.

    The cost: users running against the same conjecture are collaborating,
    not competing, and their results are entangled from the first shared
    lemma.

  * Prompt files are looked up in the conjecture directory first, then in
    the package's agents/. Nothing changes unless you drop a file in; it just means a
    conjecture that needs, say, a prover primed for Brauer groups can have one
    without forking the defaults.

  * references/ (input) and references.md (output) have different owners.
    The input directory is the conjecture's maintainer's local working copy
    of the source material — gitignored, never committed. The output,
    references.md, is the committed reference collection: the maintainer
    writes it with proofs parse and commits it with the conjecture, and every
    other user pulls it and never regenerates it. The proving loop only ever
    reads references.md; a result a user needs that is missing is requested
    from the maintainer, who parses it in (see parsing.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import settings
from ..tools.run_code import limits

CONJECTURES_ROOT = "conjectures"
# The marker file at a workspace's root, and the layout version it records.
MARKER_FILENAME = ".proofs"
# 2: append-only streams (core/streams.py). A format-1 workspace — one file
# per user and kind — is refused until proofs migrate converts it.
WORKSPACE_FORMAT = 2
CONJECTURE_FILENAME = "conjecture.md"
# The runs' lemma streams (see the module docstring).
DAGS_DIRNAME = "dags"
# The maintainer's input: a directory of .tex/.md/plain-text files to parse
# (local, gitignored — the raw sources are never committed). Its output:
# references.md, a strict-JSON array of
# { id, slogan, "formal statement", reference } objects the maintainer
# commits with the conjecture; the run feeds it to the planner, the prover
# and the verifiers, and only ever reads it.
REFERENCES_DIRNAME = "references"
REFERENCES_FILENAME = "references.md"
# Free-form operator notes on possible approaches to a proof or
# counterexample. Read once per run and handed to the planner verbatim on
# every iteration; the planner is the only agent that sees it, and a
# conjecture directory without the file runs without the section. Gitignored
# and never committed or shared: each user keeps their own copy in their
# own clone (see the README's "Comments" section).
COMMENTS_FILENAME = "comments.md"
# Every code execution of a run with --tools (execution.py): one directory
# per run under scratch/<user_id>/, local to the machine and never committed.
SCRATCH_DIRNAME = "scratch"
AGENTS_DIR = "agents"
# Every prompt the loop can load, and therefore every per-conjecture override
# that is meaningful. parsing.md is the standalone parser's (parsing.py's).
# The three verifier_*.md files are the three atomic verification steps
# (VERIFIER_AGENTS in validity/steps.py).
#
# Each entry maps the lookup name (what Paths.prompt_paths is keyed by, and what a
# per-conjecture override file is called) to the path the default lives at,
# relative to the package's agents/. The parallel-mode agents keep their own
# subdirectory on disk but stay flat in the map, so an override is always
# just "drop a file next to conjecture.md" whatever the agent is.
PROMPT_FILES: Dict[str, str] = {
    "planner.md": "planner.md",
    "prover.md": "prover.md",
    "verifier_1.md": "verifier_1.md",
    "verifier_2.md": "verifier_2.md",
    "verifier_3.md": "verifier_3.md",
    "coding_verifier.md": "coding_verifier.md",
    "reviser.md": "reviser.md",
    "coding.md": "coding.md",
    "parsing.md": "parsing.md",
    "parallel_planner.md": "parallel/parallel_planner.md",
    "parallel_lemma_generator.md": "parallel/parallel_lemma_generator.md",
}


class ConjectureNotFound(Exception):
    """Raised with the list of what *does* exist, so the fix is obvious."""


class NotInWorkspace(ConjectureNotFound):
    """A bare conjecture name, or proofs new, with no workspace to look in."""

    def __init__(self) -> None:
        super().__init__("not in a proofs workspace; run proofs init")


# ----------------------------------------------------------------------------
# The workspace: its marker, its root, and proofs init.
# ----------------------------------------------------------------------------
GITIGNORE_BEGIN = "# >>> proofs >>>"
GITIGNORE_END = "# <<< proofs <<<"
# The one source of truth for what a workspace keeps out of git: proofs init
# writes these lines between GITIGNORE_BEGIN and GITIGNORE_END in the
# workspace's .gitignore, and a re-run replaces them, so a workspace picks up
# a change here by running proofs init again. The patterns are anchored at
# the workspace root, where the .gitignore lives.
GITIGNORE_RULES = """\
# Managed by proofs init: re-running it replaces everything between these
# markers, so put your own rules outside them.
#
# Reference input material is the conjecture maintainer's local working
# copy: the maintainer points `proofs parse` at it, and the parsed output
# (references.md, beside the directory) is what gets committed with the
# conjecture. Other users pull references.md; they never see the inputs.
conjectures/*/references/
# Each user's private notes for the planner: never committed or shared,
# so every user keeps their own (README, "Comments").
conjectures/*/comments.md
.comments
# Each run's checkpoint and lock (core/checkpoint.py): local to the machine.
conjectures/*/checkpoints/
# The temp files a crashed rewrite can leave behind: proofs prune rewrites
# a stream as <file>.<pid>.tmp beside it, then os.replace; a refutation is
# written as .<name>...tmp and linked into place.
conjectures/*/dags/*/*/*.tmp
conjectures/*/certificates/*/*/*.tmp
conjectures/*/computations/*/*/*.tmp
conjectures/*/refutations/.*.tmp
conjectures/*/references.md.*.tmp
# Code tools (codetools.py, execution.py): every execution's script and
# log, and the per-machine limits — the machine's, never the conjecture's.
conjectures/*/scratch/
conjectures/*/limits.json
.proofs-local.json
conjectures/*/code/*/*/*.tmp
# Never commit API keys (README "Backend": --api-key).
.env
.env.*
"""


class NestedWorkspace(Exception):
    """Raised by init() for a directory inside an existing workspace."""


def find_root(start: Optional[Path] = None) -> Optional[Path]:
    """The workspace root: the first of start (default: the current
    directory) and its parents that holds a .proofs marker, or None."""
    start = (start or Path.cwd()).resolve()
    for d in (start, *start.parents):
        if (d / MARKER_FILENAME).is_file():
            return d
    return None


def marker_format(root: Path) -> int:
    """The format a workspace's marker records ("format = N"), 0 when it has
    no readable one."""
    try:
        text = (root / MARKER_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return 0
    m = re.search(r"^\s*format\s*=\s*(\d+)\s*$", text, re.M)
    return int(m.group(1)) if m else 0


class OldFormat(ConjectureNotFound):
    """A conjecture still in format 1: proofs migrate converts it."""


def _in_git_repo(path: Path) -> bool:
    """Whether path is inside a git work tree: a .git (directory, or the
    file a worktree or submodule has) in it or a parent. git is not run."""
    return any((d / ".git").exists() for d in (path, *path.parents))


def gitignore_block() -> str:
    """The proofs block, markers included, as init() writes it."""
    return f"{GITIGNORE_BEGIN}\n{GITIGNORE_RULES}{GITIGNORE_END}\n"


def update_gitignore(text: str) -> str:
    """text, a .gitignore's contents, with its proofs block replaced in place
    by gitignore_block(), or the block appended when it has none. Lines
    outside the block are never touched."""
    lines = text.splitlines(keepends=True)
    stripped = [l.rstrip("\r\n").strip() for l in lines]
    try:
        begin = stripped.index(GITIGNORE_BEGIN)
        end = stripped.index(GITIGNORE_END, begin + 1)
    except ValueError:
        if text and not text.endswith("\n"):
            text += "\n"
        sep = "\n" if text.strip() else ""
        return (text if text.strip() else "") + sep + gitignore_block()
    return "".join(lines[:begin]) + gitignore_block() + "".join(lines[end + 1:])


def init(directory: Path) -> Tuple[Path, bool]:
    """Lay out a proofs workspace at directory, for proofs init.

    Creates directory and its conjectures/, writes the .proofs marker and
    a .proofs-local.json holding the built-in code limits and run settings
    (limits.DEFAULTS, settings.DEFAULTS) when it is missing, and writes or updates the proofs block of its
    .gitignore. Safe to re-run: an existing marker, limits file and
    conjectures are left alone, and only the block is rewritten. A directory inside
    another workspace is refused — a nested workspace is almost certainly a
    mistake. Returns the root and whether it is inside a git repository.
    """
    root = directory.resolve()
    outer = find_root(root)
    if outer is not None and outer != root:
        raise NestedWorkspace(
            f"{root} is inside the proofs workspace at {outer}; "
            f"a workspace cannot be nested in another."
        )
    (root / CONJECTURES_ROOT).mkdir(parents=True, exist_ok=True)
    marker = root / MARKER_FILENAME
    if not marker.is_file():
        marker.write_text(f"format = {WORKSPACE_FORMAT}\n", encoding="utf-8")
    local = root / limits.LOCAL_FILENAME
    if not local.exists():
        local.write_text(
            json.dumps({**limits.DEFAULTS, **settings.DEFAULTS}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    gitignore = root / ".gitignore"
    old = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    new = update_gitignore(old)
    if new != old:
        gitignore.write_text(new, encoding="utf-8")
    return root, _in_git_repo(root)


@dataclass
class RunPaths:
    root: Path                  # conjectures/algebra_example
    conjecture: Path            # conjectures/algebra_example/conjecture.md
    dags_dir: Path              # conjectures/algebra_example/dags/ (every run's lemma stream)
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


def create(name: str, project_root: Optional[Path]) -> Path:
    """Lay out a blank conjecture, <project_root>/conjectures/<name>/, for
    proofs new. project_root is the workspace root (find_root()); None, no
    workspace, raises NotInWorkspace.

    The input directory references/ and the files: conjecture.md and comments.md empty, references.md the
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
    if project_root is None:
        raise NotInWorkspace()
    root = project_root / CONJECTURES_ROOT / name
    if root.exists():
        raise ConjectureExists(f"{root} already exists.")
    root.mkdir(parents=True)
    (root / REFERENCES_DIRNAME).mkdir()
    (root / CONJECTURE_FILENAME).write_text("", encoding="utf-8")
    (root / COMMENTS_FILENAME).write_text("", encoding="utf-8")
    (root / REFERENCES_FILENAME).write_text("[]\n", encoding="utf-8")
    return root


def available(project_root: Optional[Path]) -> List[str]:
    """The conjectures under the workspace root's conjectures/; none when
    project_root is None (no workspace)."""
    if project_root is None:
        return []
    root = project_root / CONJECTURES_ROOT
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / CONJECTURE_FILENAME).is_file()
    )


def is_bare_name(name: str) -> bool:
    """Whether name is a bare conjecture name (looked up under the
    workspace's conjectures/) rather than a path: a single component, a
    trailing separator aside, that is not . or .."""
    stripped = name.rstrip("/\\")
    return (
        bool(stripped)
        and "/" not in stripped and "\\" not in stripped
        and stripped not in (".", "..")
    )


def agents_dir() -> Path:
    """The agents/ directory of default prompts shipped inside this package.

    The prompts are package data (pyproject.toml's package-data), installed
    beside the code that parses their output, so every install — pip, Nix,
    or a checkout — runs with the prompts its own version expects.
    """
    return Path(__file__).resolve().parents[1] / AGENTS_DIR


def resolve(
    name: str,
    user_id: str,
    project_root: Optional[Path] = None,
) -> RunPaths:
    """Turn a conjecture name and the run's user into every path the run needs.

    `name` is accepted in whichever form is convenient:
        algebra_example
        conjectures/algebra_example
        conjectures/algebra_example/          (trailing slash, from tab-completion)
        /abs/path/to/algebra_example

    A path — anything with a directory separator, or . or .. — is taken
    relative to the current directory, or as given when absolute. A bare
    name is looked up under <project_root>/conjectures/, project_root being
    the workspace root (find_root()); with no workspace (None), a bare name
    raises NotInWorkspace. The paths returned are absolute, so nothing
    downstream depends on the directory the command was run from.

    A prompt is taken from the conjecture directory when a file of that
    name is there, else from the defaults shipped in the package — see
    agents_dir().
    """
    stripped = name.rstrip("/\\") or name
    if is_bare_name(name):
        if project_root is None:
            raise NotInWorkspace()
        guess = project_root / CONJECTURES_ROOT / stripped
    else:
        guess = Path(stripped)
    if guess.is_dir():
        root = guess.resolve()
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
    workspace_root = find_root(root)
    if (workspace_root is not None and marker_format(workspace_root) < WORKSPACE_FORMAT) \
            or _has_format1_files(root):
        where = workspace_root or root.parent.parent
        raise OldFormat(
            f"{root} is laid out in format 1 (one file per user); run "
            f"proofs migrate {where} to convert the workspace (commit your "
            f"work first, so the change can be reviewed)"
        )

    dags_dir = root / DAGS_DIRNAME

    agents = agents_dir()
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
                    references_dir=root / REFERENCES_DIRNAME,
                    references=root / REFERENCES_FILENAME,
                    comments=root / COMMENTS_FILENAME,
                    prompts=prompts, overridden_prompts=overridden)


def _has_format1_files(root: Path) -> bool:
    """Whether a conjecture still holds format 1's per-user files (the
    cheap check; core.migrate.old_files finds every one)."""
    return (any((root / DAGS_DIRNAME).glob("*_dag.json"))
            or any(p.is_file() for p in (root / "certificates").glob("*.jsonl"))
            or any((root / "runs").glob("*.jsonl")))


def describe(paths: RunPaths) -> str:
    lines = [f"conjecture={paths.root}"]
    if paths.dags_dir.is_dir():
        runs = [r for user in paths.dags_dir.iterdir() if user.is_dir()
                for r in user.iterdir() if r.is_dir()]
        users = {r.parent.name for r in runs}
        lines.append(
            f"  dags/: {len(runs)} run(s) by {len(users)} user(s)"
            if runs else "  dags/: no lemmas yet"
        )
    if paths.references_dir.is_dir():
        n_files = sum(1 for p in paths.references_dir.iterdir() if p.is_file())
        lines.append(
            f"  references={paths.references_dir.name}/ ({n_files} file(s) for "
            f"proofs parse)"
        )
    if paths.references.is_file():
        try:
            entries = json.loads(paths.references.read_text(encoding="utf-8"))
            n = len(entries) if isinstance(entries, list) else -1
        except (OSError, json.JSONDecodeError):
            n = -1
        if n < 0:
            lines.append(
                f"  references={paths.references.name} is not valid JSON; "
                f"the loop will ignore it"
            )
        else:
            lines.append(f"  references={paths.references.name} ({n} entries)")
    comments = paths.comments
    if comments.is_file() and comments.read_text(encoding="utf-8").strip():
        lines.append(
            f"  comments={comments.name} passed to the planner verbatim as suggestions"
        )
    if paths.overridden_prompts:
        lines.append(
            f"  local prompt overrides: {', '.join(paths.overridden_prompts)}"
        )
    return "\n".join(lines)
