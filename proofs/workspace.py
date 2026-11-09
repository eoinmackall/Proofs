"""The proofs workspace and its per-conjecture run directories.

A workspace is a directory holding a .proofs marker file, laid out by
proofs init (init()); every command finds it by walking up from the current
directory to the first one with the marker (find_root()), the way git finds
.git. Conjecture paths below are relative to that root.

Layout:

    .proofs                        <- the marker: "format = 1"
    .proofs-local.json             <- optional, gitignored: this machine's code limits (limits.py)
    .gitignore                     <- holds the proofs block (GITIGNORE_RULES)
    conjectures/
      algebra_example/
        conjecture.md              <- required, the only mandatory file
        dags/                      <- committed: per-user DAG files, <user_id>_dag.json,
                                      and (gitignored) each user's <user_id>_dag.checkpoint.json
        certificates/              <- committed: per-verifier acceptances, <user_id>.jsonl
        refutations/               <- committed: one file per rejection, created on first write
        references/                <- maintainer-local inputs for proofs parse (gitignored)
        references.md              <- committed: strict-JSON refs, proofs parse's output
        comments.md                <- optional, gitignored: this user's notes, passed to the planner
        prover.md                  <- optional per-conjecture prompt override
        flake.nix, flake.lock      <- optional, committed: the maintainer's code tools
        tools.lock                 <- committed with them: the tools runs may use (codetools.py)
        limits.json                <- optional, gitignored: this machine's limits here (limits.py)
        scratch/<user_id>/         <- gitignored: every code execution's script and log
        code/<user_id>/            <- committed: the scripts cited proofs rest on (computations.py)
        runs/<user_id>.jsonl       <- committed: that user's executions of them
      some_other_problem/
        conjecture.md

    proofs/agents/                 <- inside the package, installed with it
      planner.md  prover.md  verifier_1/2/3.md  reviser.md  <- default prompts

Three decisions worth flagging, all changeable:

  * One DAG file per user, not per model or per run:
    conjectures/NAME/dags/<user_id>_dag.json. load_dag() merges every file
    in dags/ into the complete DAG each time it is called, so a user who
    pulls an existing conjecture picks up wherever everyone else stopped,
    whatever wrote those lemmas: the proof is the artefact, and a lemma one
    person proved and certified is no less proved when another person
    arrives to continue. Each lemma stores the user_id and lemma_id that
    identify it, and a lemma_id is unique only within one user's file —
    which is why the merge is keyed by the pair (user_id, lemma_id). A run
    writes new lemmas to the current user's file only, never to another
    user's, so users exchanging files through git rarely edit the same one
    (proofs repair's in-place rewrite of a lemma is the exception).

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

try:  # inside the proofs package (installed, or `python -m proofs`)
    from . import certificates
except ImportError:  # top-level modules (`python proofs/workspace.py`)
    import certificates

CONJECTURES_ROOT = "conjectures"
# The marker file at a workspace's root, and the layout version it records.
MARKER_FILENAME = ".proofs"
WORKSPACE_FORMAT = 1
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
# (VERIFIER_AGENTS in main.py).
#
# Each entry maps the lookup name (what PROMPT_PATHS is keyed by, and what a
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
*.checkpoint.json
# Per-writer temp files a crashed checkpoint write can leave behind
# (named <checkpoint>.<pid>.<thread>.tmp).
*.checkpoint.json.*.tmp
# Cross-process lock sidecars beside the DAG and certificate files
# (locking.py), and the temp files a crashed write can leave behind: every
# writer of a committed file writes <file>...tmp beside it, then
# os.replace (or os.link, for a refutation).
conjectures/*/dags/*.lock
conjectures/*/certificates/*.lock
conjectures/*/dags/*.tmp
conjectures/*/certificates/*.tmp
conjectures/*/refutations/.*.tmp
conjectures/*/references.md.*.tmp
# Code tools (codetools.py, execution.py): every execution's script and
# log, and the per-machine limits — the machine's, never the conjecture's.
conjectures/*/scratch/
conjectures/*/limits.json
.proofs-local.json
conjectures/*/runs/*.lock
conjectures/*/runs/*.tmp
conjectures/*/code/*/*.tmp
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

    Creates directory and its conjectures/, writes the .proofs marker when
    it is missing, and writes or updates the proofs block of its
    .gitignore. Safe to re-run: an existing marker and existing conjectures
    are left alone, and only the block is rewritten. A directory inside
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


def create(name: str, project_root: Optional[Path]) -> Path:
    """Lay out a blank conjecture, <project_root>/conjectures/<name>/, for
    proofs new. project_root is the workspace root (find_root()); None, no
    workspace, raises NotInWorkspace.

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
    if project_root is None:
        raise NotInWorkspace()
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
    return Path(__file__).resolve().parent / AGENTS_DIR


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

    The run's DAG file is dags/<user_id>_dag.json in the conjecture directory:
    the user's share of the conjecture's proof. The run reads every file in
    dags/ (the complete DAG) and writes this file alone.

    A prompt is taken from the conjecture directory when a file of that
    name is there, else from the defaults shipped in the package — see
    agents_dir().
    """
    if not user_id:
        raise ValueError("resolve() needs a user_id to name the run's DAG file")
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

    dags_dir = root / DAGS_DIRNAME
    dag = dags_dir / (user_id + USER_DAG_SUFFIX)

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
            f"  resuming from existing {paths.dag.name} "
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
                f"  other users' DAG files in {paths.dags_dir.name}/: "
                f"{', '.join(others)} — merged into the run, never written"
            )
    checkpoint = paths.dag.with_name(
        paths.dag.stem + ".checkpoint" + paths.dag.suffix
    )
    if checkpoint.exists():
        lines.append(
            f"  checkpoint {checkpoint.name} present — the run resumes "
            f"from it (--fresh to restart the budget)"
        )
    # Runs from before DAGs were per-user left dag.json and
    # dag-<backend>-<model>.json files here. They are not read any more, and
    # saying so beats letting someone conclude the lemmas were lost.
    legacy_shared = paths.root / LEGACY_DAG_FILENAME
    if legacy_shared.is_file():
        lines.append(
            f"  ignoring {legacy_shared.name} — the pre-per-user shared "
            f"DAG; move it to {paths.dags_dir.name}/<user_id>{USER_DAG_SUFFIX} "
            f"to carry its lemmas into the run"
        )
    legacy = sorted(p.name for p in paths.root.glob("dag-*.json"))
    if legacy:
        lines.append(
            f"  ignoring per-model DAG{'s' if len(legacy) > 1 else ''} "
            f"{', '.join(legacy)} — the per-user DAGs in {paths.dags_dir.name}/ "
            f"replaced them"
        )
    # Checkpoints now live beside the user's DAG file in dags/, so a
    # *.checkpoint.json at the conjecture root is an orphan of the shared
    # DAG: nothing reads it, and saying so beats it looking live.
    legacy_cp = sorted(p.name for p in paths.root.glob("*.checkpoint.json"))
    if legacy_cp:
        lines.append(
            f"  ignoring stale checkpoint{'s' if len(legacy_cp) > 1 else ''} "
            f"{', '.join(legacy_cp)} — checkpoints now live beside the "
            f"user's DAG file in {paths.dags_dir.name}/"
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


# ----------------------------------------------------------------------------
# Ownership: whether the run's user is the user who owns the files its
# user_id names.
# ----------------------------------------------------------------------------
# user.id names a user's files and nothing enforces that it is unique: two
# people who pick the same id would write one DAG file and one certificate
# file between them, each run rewriting the header with its own name and
# email. One person on two machines is fine — same id, same email — so the
# identity stamped at the top of the files (see config.py) is the test:
# email decides when the file records one, name when it records none, and a
# file that records neither (or has no header, from before headers) is
# taken as the run's.

class OwnerConflict(Exception):
    """The user's files carry another person's identity."""


def _stored_identity(path: Path) -> Optional[Dict[str, str]]:
    """The identity header of a user's DAG file (top-level keys) or
    certificate file (first line), or None when there is none to read."""
    if not path.is_file():
        return None
    try:
        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                data = json.loads(f.readline() or "null")
            if not certificates.is_header(data):
                return None
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # unreadable: the writers report it themselves
    if not isinstance(data, dict):
        return None
    return {
        k: str(data.get(k) or "").strip()
        for k in (certificates.USER_NAME_KEY, certificates.USER_EMAIL_KEY)
    }


def owner_files(root: Path, user_id: str) -> List[Path]:
    """The files user_id names in the conjecture at root, the ones that
    carry its owner's identity: dags/<user_id>_dag.json and
    certificates/<user_id>.jsonl."""
    return [
        root / DAGS_DIRNAME / (user_id + USER_DAG_SUFFIX),
        certificates.file_for(root, user_id),
    ]


def owner_conflict(
    root: Path, user_id: str, name: str, email: str
) -> Optional[str]:
    """Why the user with this name and email may not write user_id's files
    in the conjecture at root, or None when they may.

    Emails are compared ignoring case and surrounding space. A file that
    records an email conflicts with a config that holds a different one, or
    none — an unset email cannot show it is the same person. A file that
    records only a name conflicts with a different name."""
    name, email = name.strip(), email.strip()
    for path in owner_files(root, user_id):
        stored = _stored_identity(path)
        if stored is None:
            continue
        s_name = stored[certificates.USER_NAME_KEY]
        s_email = stored[certificates.USER_EMAIL_KEY]
        if s_email:
            if s_email.lower() == email.lower():
                continue
            mine = f"user.email {email!r}" if email else "no user.email"
            theirs = f"user.email {s_email!r}"
        elif s_name and name and s_name != name:
            mine, theirs = f"user.name {name!r}", f"user.name {s_name!r}"
        else:
            continue
        rel = path.relative_to(root)
        return (
            f"user.id {user_id!r} is already taken in {root.name}: "
            f"{rel} belongs to {s_name or 'someone'} ({theirs}), and your "
            f"config has {mine}.\n"
            f"Each person needs their own user.id — set one with "
            f"proofs config --global user.id ID.\n"
            f"If that file is yours under an old name or email, set yours to "
            f"match (proofs config --global user.email ...), or edit the "
            f"user.name and user.email at the top of "
            f"{user_id}{USER_DAG_SUFFIX} and on the first line of "
            f"{user_id}.jsonl."
        )
    return None


def check_owner(paths: RunPaths, user_id: str, name: str, email: str) -> None:
    """Raise OwnerConflict when owner_conflict finds one."""
    reason = owner_conflict(paths.root, user_id, name, email)
    if reason is not None:
        raise OwnerConflict(reason)
