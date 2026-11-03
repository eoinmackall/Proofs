"""The `proofs` command: one entry point for every proofs tool.

    proofs config KEY [VALUE]         get or set the user's config, git-style
    proofs init [DIR]                 make DIR (default: .) a proofs workspace
    proofs new NAME                   lay out a blank conjecture, conjectures/NAME/
    proofs parse DIR                  parse DIR's references/ into references.md
    proofs run DIR                    run the proof loop on DIR
    proofs verify PATH lemma_id       verify one lemma of a user's DAG [--full]
    proofs repair path_to_refutation  repair the lemma a refutation file names
    proofs prune DIR                  drop the current user's stale certificates
    proofs status PATH lemma_id       list the valid certificates in a lemma's closure
    proofs export PATH [lemma_id] -o OUT_DIR
                                    write a LaTeX document of the DAG

A workspace is the directory proofs init marks with a .proofs file, found
by walking up from the current directory (workspace.find_root); main()
finds it once and hands it to every command that needs it, as
args.workspace_root. conjectures/NAME below is relative to that root.

DIR is a conjecture directory: a directory holding a conjecture.md, given
as a bare NAME (looked up in the workspace's conjectures/) or as a path,
relative to the current directory or absolute. PATH is the path to a
user's DAG file — conjectures/NAME/dags/<user_id>_dag.json — which supplies
the lemma's user_id, so a lemma is identified by the pair (user_id,
lemma_id). For export, PATH may also be the conjecture directory itself, in
which case the complete DAG is written. A path works outside a workspace; a
bare NAME, and proofs new, need one.

This module validates the arguments and hands each command to its entry
point, where the command's own docstring says what it does:

  * parse and run forward to parsing.run() and main.main() with DIR as
    --conjecture, and pass anything after DIR (--model, --mode,
    --max-iterations, --parallel, ...) through unchanged.
  * verify (main.verify_entry) runs the three verifier checks on a lemma
    that is already in the DAG: a rejection writes a refutation file
    (refutations.py), an acceptance a certificate (certificates.py), and
    the lemma's stale refutations give their justifications to the
    verifiers either way.
  * repair (main.repair_entry) is addressed to one refutation file by its
    path. It re-verifies the lemma with the refutation's justification as
    input, deleting the file when the proof is accepted; a rejected proof
    is re-proven through the prover, verifier and reviser, committed in
    place under the lemma's own user_id and lemma_id, and the lemmas that
    depend on it are re-verified up the chain, each re-proved if its
    re-verification rejects it.
  * prune (main.prune_entry) recomputes the Merkle hash of every lemma of
    the conjecture and drops the current user's certificates that no
    longer match — the ones that user issued as verifier — never touching
    another user's file.
  * status (main.status_entry) reports, for each lemma in the named
    lemma's dependency closure, every valid certificate — a line of any
    user's certificate file whose hash matches the lemma's current Merkle
    hash — with its verifier, model, date and count. It writes nothing.
  * export (main.export_entry, export.py) writes the DAG as a LaTeX
    document in dependency order — with a user's DAG file, that user's
    lemmas, and with a lemma id, that lemma and everything it cites; with
    the conjecture directory, the complete DAG — to proof.tex,
    <user_id>_proof.tex or <user_id>--<lemma_id>_proof.tex in the
    directory -o/--out names, created if it is not there yet. -o/--out is
    required: the document is the user's artifact, not the project's
    data, so it never lands in the project on its own.
  * init (workspace.init) creates DIR/conjectures/, writes the .proofs
    marker and writes or updates the proofs block of DIR/.gitignore. It
    is safe to re-run, and refuses a DIR inside another workspace.
  * new creates conjectures/NAME/ under the workspace root with empty
    references/, dags/ and certificates/ directories, an empty
    conjecture.md and comments.md, and a references.md holding the empty
    collection []. It refuses a NAME that is already there.

None of the commands run git: users share a conjecture's files by
committing, pushing and pulling them by hand. Every file a command writes
belongs to the user running it (their DAG file, their certificate file, a
refutation named for them), so two users' work never lands in the same
file. The exceptions are verify and repair: an acceptance deletes the
lemma's refutations whoever wrote them, and repair rewrites the repaired
lemma in its owner's DAG file.

Every command except `config`, `init` and `new` acts as the user the config's
user.id names (see config.py): the id is read from the config, and a
run without one fails and names the command that sets it — proofs does not
ask, as git does not. The config lives in ~/.config/proofs/config, outside
the repository, and is never committed; its keys (user.id, user.name,
user.email) are set and read through the config subcommand, git-style:
`proofs config --global user.id ID` sets, `proofs config user.id` prints,
an empty value unsets, and `--list` lists. The resolved id is left on args
as args.user_id.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # inside the proofs package: installed, or `python -m proofs`
    from . import config
    from . import main as main_mod
    from . import parsing as parsing_mod
    from . import workspace
except ImportError:  # top-level modules: `python proofs/cli.py`
    import config
    import main as main_mod
    import parsing as parsing_mod
    import workspace

# Argument errors exit 2, argparse's convention; a command's own exit
# code is what its entry returns (0 accepted, 1 rejected, and so on).

# A user's DAG file is named <user_id>_dag.json, and the name is the
# source of the lemma's user_id: load_dag takes the owner of every lemma
# it reads from the file name, so the name and the lemma agree by
# construction, and the file's contents need no owner field of their own.
_USER_DAG_NAME = re.compile(r"(?P<user_id>.+)_dag\.json")


def _add_backend_flags(sp: argparse.ArgumentParser, help_for: str) -> None:
    """The model and transport flags verify and repair take, the run's
    (main.py's parser, shortened): which model under which backend the
    work runs, and the context ceiling. help_for names the work in the
    --verbose help ("the verification", "the re-verification")."""
    sp.add_argument(
        "--model", required=True,
        help="the model the work runs under (required)",
    )
    sp.add_argument(
        "--host", required=True,
        help="OpenAI-compatible base URL (required)",
    )
    sp.add_argument(
        "--api-key",
        default=None,
        help="the backend's API key (default: none sent)",
    )
    sp.add_argument(
        "--backend",
        choices=["auto", "llamacpp", "openai"],
        default="auto",
        help="which backend to use (default: auto)",
    )
    sp.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="context window in tokens; if the server has a lower ceiling, the ceiling wins",
    )
    sp.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"log {help_for} to stdout (default: on; --no-verbose silences it)",
    )


_DIR_HELP = (
    "The conjecture directory: a NAME under the workspace root's "
    "conjectures/, or a path to it."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proofs",
        description=(
            "Distributed proof construction, coordinated through git. "
            "DIR is a conjecture directory: a NAME under the workspace "
            "root's conjectures/, or a path to it. PATH is the path to a "
            "user's DAG file, which supplies the lemma's user_id (for "
            "export it may also be a conjecture directory). Run proofs "
            "init to make a workspace."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    p = sub.add_parser(
        "config",
        help="Get or set a key of the user's config, git-style.",
        description=(
            "Get or set a key of the user's config, git-style, as git "
            "config's are. proofs config KEY VALUE sets, proofs config KEY "
            "prints, and an empty value unsets. The keys are user.id, "
            "user.name and user.email; the config lives in "
            "~/.config/proofs/config, outside the repository, and is never "
            "committed."
        ),
    )
    p.add_argument(
        "--global",
        dest="scope",
        action="store_true",
        default=False,
        help=(
            "The global config (~/.config/proofs/config, never committed) — "
            "the only config proofs has, accepted as git accepts it."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List every set key, as key=value."
    )
    p.add_argument(
        "key", nargs="?", default=None, metavar="key",
        help="user.id, user.name or user.email.",
    )
    p.add_argument(
        "value", nargs="?", default=None, metavar="value",
        help="The value to set; an empty value unsets the key.",
    )
    p.set_defaults(func=_cmd_config, parser=p)

    p = sub.add_parser(
        "init",
        help="Make DIR (default: the current directory) a proofs workspace.",
        description=(
            "Make DIR a proofs workspace: create DIR/conjectures/, write "
            "the .proofs marker, and write or update the proofs block of "
            "DIR/.gitignore (lines outside the block are left alone). Safe "
            "to re-run. A DIR inside another workspace is refused."
        ),
    )
    p.add_argument(
        "dir", nargs="?", default=".", metavar="DIR",
        help="The workspace directory, created if it is not there yet "
             "(default: the current directory).",
    )
    p.set_defaults(func=_cmd_init, parser=p)

    p = sub.add_parser(
        "new",
        help="Lay out a blank conjecture, conjectures/NAME/ under the "
             "workspace root.",
        description=(
            "Create conjectures/NAME/, relative to the workspace root, with "
            "empty references/, dags/ and "
            "certificates/ directories, an empty conjecture.md and "
            "comments.md, and a references.md holding the empty collection "
            "[]. An existing NAME is refused. Write the statement into "
            "conjecture.md before proofs run."
        ),
    )
    p.add_argument(
        "name", metavar="NAME",
        help="The conjecture's directory name, e.g. standard_d.",
    )
    p.set_defaults(func=_cmd_new, parser=p)

    p = sub.add_parser(
        "parse",
        help="Parse DIR's references/ directory into references.md.",
        description=(
            "Parse the conjecture's references/ directory into "
            "references.md. Anything after DIR is passed to the parser "
            "unchanged (--model, --host, --num-ctx, ...)."
        ),
    )
    p.add_argument("dir", metavar="DIR", help=_DIR_HELP)
    # Everything after DIR: the parser's own flags, validated by its own
    # parser (so their help and errors are the parser's, not ours).
    p.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_parse, parser=p)

    p = sub.add_parser(
        "run",
        help="Run the proof loop on DIR.",
        description=(
            "Run the multi-agent proof loop on a conjecture. Anything after "
            "DIR is passed to the loop unchanged (--model, --mode, "
            "--max-iterations, --parallel, ...)."
        ),
    )
    p.add_argument("dir", metavar="DIR", help=_DIR_HELP)
    p.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_run, parser=p)

    p = sub.add_parser(
        "verify",
        help="Verify one lemma of a user's DAG file.",
        description=(
            "Verify the named lemma of a user's DAG file. With --full, also "
            "verify everything it cites, cited lemmas first. There is no "
            "mode that verifies every lemma at once."
        ),
    )
    p.add_argument("path", metavar="PATH", help="The user's DAG file.")
    p.add_argument("lemma_id", help="The lemma to verify.")
    p.add_argument(
        "--full",
        action="store_true",
        help="Also verify everything the lemma cites, cited lemmas first.",
    )
    _add_backend_flags(p, help_for="the verification")
    p.set_defaults(func=_cmd_verify, parser=p)

    p = sub.add_parser(
        "repair",
        help="Repair the lemma a refutation file names.",
        description=(
            "Repair the lemma a refutation file names. The refutation file "
            "supplies the lemma's user_id and lemma_id, so no other "
            "arguments are needed."
        ),
    )
    p.add_argument(
        "path",
        metavar="path_to_refutation",
        help="A file in conjectures/NAME/refutations/ (under the "
             "workspace root).",
    )
    _add_backend_flags(p, help_for="the re-verification")
    main_mod.add_lemma_window_flags(p)
    main_mod.add_tool_flags(p)
    p.set_defaults(func=_cmd_repair, parser=p)

    p = sub.add_parser(
        "prune",
        help="Drop the current user's stale certificates in DIR.",
        description=(
            "Recompute all of DIR's hashes and delete the current user's "
            "certificates that no longer match. Other users' certificates "
            "are never touched."
        ),
    )
    p.add_argument("dir", metavar="DIR", help=_DIR_HELP)
    p.set_defaults(func=_cmd_prune, parser=p)

    p = sub.add_parser(
        "status",
        help="List the valid certificates in a lemma's closure.",
        description=(
            "For each lemma in the named lemma's citation closure, list "
            "every valid certificate, with its verifier, model, date and "
            "count."
        ),
    )
    p.add_argument("path", metavar="PATH", help="The user's DAG file.")
    p.add_argument("lemma_id", help="The lemma whose closure to list.")
    p.set_defaults(func=_cmd_status, parser=p)

    p = sub.add_parser(
        "export",
        help="Write a LaTeX document of the DAG, in dependency order.",
        description=(
            "Write the DAG as a LaTeX document, in dependency order "
            "(dependencies first). With a user's DAG file, its lemmas; with "
            "a lemma id, that lemma and everything it cites. With the "
            "conjecture directory, the complete DAG. The document is "
            "written into the -o/--out directory (created if it is not "
            "there yet), named proof.tex, <user_id>_proof.tex or "
            "<user_id>--<lemma_id>_proof.tex."
        ),
    )
    p.add_argument(
        "path", metavar="PATH",
        help="A user's DAG file, or the conjecture directory.",
    )
    p.add_argument(
        "lemma_id", nargs="?", default=None,
        help="Optional: with a DAG file, write this lemma and everything "
             "it cites.",
    )
    p.add_argument(
        "-o", "--out", dest="out", required=True, metavar="OUT_DIR",
        help="The directory the .tex file is written to, created if it is "
             "not there yet. The document never lands in the project on "
             "its own.",
    )
    p.set_defaults(func=_cmd_export, parser=p)
    return parser


# ----------------------------------------------------------------------------
# Argument validation: the shapes DIR and PATH must have before a command
# can do anything with them.
# ----------------------------------------------------------------------------
def _require_conjecture_dir(
    dir_arg: str, user_id: str, root: Optional[Path],
    parser: argparse.ArgumentParser,
) -> Path:
    """DIR must be a conjecture directory; resolve it for its root path.

    Accepted in whatever form is convenient (algebra_example,
    conjectures/algebra_example, /abs/path), as workspace.resolve() does;
    a bare name is looked up under the workspace root, and is an error
    without one.
    """
    try:
        return workspace.resolve(dir_arg, user_id, root).root
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))


def _load_user_dag(
    path: str, parser: argparse.ArgumentParser
) -> Dict[str, Any]:
    """PATH must be a readable user's DAG file: a JSON object with a
    "lemmas" object."""
    p = Path(path)
    if not p.is_file():
        parser.error(
            f"{path} is not a file; PATH must be a user's DAG file "
            f"(conjectures/NAME/dags/<user_id>_dag.json, under the "
            f"workspace root)"
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        parser.error(f"{path} is not a readable DAG file: {e}")
    if not isinstance(data, dict) or not isinstance(data.get("lemmas"), dict):
        parser.error(f"{path} is not a DAG file (no \"lemmas\" object)")
    return data


def _lemma(
    dag: Dict[str, Any], path: str, lemma_id: str,
    parser: argparse.ArgumentParser,
) -> Dict[str, Any]:
    """The lemma record PATH's DAG holds under lemma_id, or an error."""
    lemmas = dag["lemmas"]
    if lemma_id not in lemmas:
        parser.error(f"{lemma_id} is not a lemma of {path}")
    node = lemmas[lemma_id]
    if not isinstance(node, dict):
        parser.error(f"{lemma_id} in {path} is malformed (not a JSON object)")
    return node


# ----------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------
def _cmd_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """The config subcommand: get and set user identity keys, git-style.

    Get and set are silent successes, as git config's are; a get of a key
    that is not set prints nothing and exits 1, as git's does. A config file
    that is not a JSON object is a one-line error, the way every other
    command reports it, not a traceback.
    """
    try:
        _config_command(args, parser)
    except config.UserConfigError as e:
        print(f"proofs: {e}", file=sys.stderr)
        sys.exit(1)


def _config_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.list and args.key is not None:
        parser.error("--list takes no key")
    if args.list:
        for key, value in config.items().items():
            print(f"{key}={value}")
        return
    if args.key is None:
        parser.error(
            "no key given; usage: proofs config [--global] KEY [VALUE], "
            "where KEY is user.id, user.name or user.email"
        )
    try:
        if args.value is None:  # a bare key is a get
            value = config.get(args.key)
            if value is None:
                sys.exit(1)  # git: a missing key prints nothing, exits 1
            print(value)
            return
        config.set(args.key, args.value)  # an empty value unsets
    except ValueError as e:
        parser.error(str(e))
    _warn_owner_conflicts(_workspace_root(warn=False))


def _warn_owner_conflicts(project_root: Optional[Path]) -> None:
    """After an identity key is set, warn about every conjecture in the
    workspace at project_root whose files under the config's user.id carry
    another person's name and email (workspace.owner_conflict). A warning,
    not an error: the config is global and this workspace is only one place
    the id is used; the commands that write the files refuse on their own.
    Outside a workspace (None) there is nothing to check."""
    uid = config.get(config.USER_ID_KEY)
    if uid is None or project_root is None:
        return
    name = config.get(config.USER_NAME_KEY) or ""
    email = config.get(config.USER_EMAIL_KEY) or ""
    for conj in workspace.available(project_root):
        root = project_root / workspace.CONJECTURES_ROOT / conj
        reason = workspace.owner_conflict(root, uid, name, email)
        if reason is not None:
            print(f"proofs: warning: {reason}", file=sys.stderr)


def _workspace_root(warn: bool = True) -> Optional[Path]:
    """The workspace root for this invocation: workspace.find_root() from
    the current directory, or None outside a workspace.

    During the transition to marked workspaces, a directory with no marker
    above it but a ./conjectures/ of its own is taken as the root, with a
    warning (when warn) that proofs init should be run there. main() calls
    this once, so the warning is printed once.
    """
    root = workspace.find_root()
    if root is not None:
        return root
    cwd = Path.cwd()
    if (cwd / workspace.CONJECTURES_ROOT).is_dir():
        if warn:
            print(
                f"proofs: warning: no {workspace.MARKER_FILENAME} marker "
                f"found; using ./{workspace.CONJECTURES_ROOT}/ in the current "
                f"directory. Run proofs init here to make it a workspace.",
                file=sys.stderr,
            )
        return cwd.resolve()
    return None


def _cmd_init(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        root, in_git = workspace.init(Path(args.dir))
    except workspace.NestedWorkspace as e:
        parser.error(str(e))
    except OSError as e:
        print(f"proofs: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Initialized proofs workspace in {root}")
    if not in_git:
        print(
            "Hint: Not inside a git repository. To share this workspace, "
            "run `git init`"
        )


def _cmd_new(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        root = workspace.create(args.name, args.workspace_root)
    except (ValueError, workspace.ConjectureExists,
            workspace.NotInWorkspace) as e:
        parser.error(str(e))
    print(
        f"Created {root}/: conjecture.md, comments.md, references.md, "
        f"references/, dags/, certificates/.\n"
        f"Write the statement into {root / workspace.CONJECTURE_FILENAME}, "
        f"then: proofs run {root}"
    )


def _require_workspace_for_name(
    dir_arg: str, root: Optional[Path], parser: argparse.ArgumentParser
) -> None:
    """A bare NAME needs a workspace to look in. Checked before run and
    parse hand off, so the error comes before their backend probe."""
    if root is None and workspace.is_bare_name(dir_arg):
        parser.error(str(workspace.NotInWorkspace()))


def _cmd_parse(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _require_workspace_for_name(args.dir, args.workspace_root, parser)
    parsing_mod.run(
        argv=["--conjecture", args.dir, *args.rest], prog="proofs parse",
        root=args.workspace_root,
    )


def _cmd_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _require_workspace_for_name(args.dir, args.workspace_root, parser)
    main_mod.main(
        argv=["--conjecture", args.dir, *args.rest], prog="proofs run",
        root=args.workspace_root,
    )


def _cmd_verify(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # Resolved first: PATH is relative to the current directory or
    # absolute, and the conjecture root is found by climbing from it — a
    # bare relative name (alice_dag.json, run inside dags/) has no parents
    # to climb until it is made absolute.
    p = Path(args.path).resolve()
    dag = _load_user_dag(args.path, parser)
    _lemma(dag, args.path, args.lemma_id, parser)
    m = _USER_DAG_NAME.fullmatch(p.name)
    if m is None:
        parser.error(
            f"{args.path} is not named <user_id>_dag.json; PATH must be a "
            f"user's DAG file (conjectures/NAME/dags/<user_id>_dag.json), "
            f"whose name supplies the lemma's user_id"
        )
    rc = main_mod.verify_entry(
        path=args.path,
        lemma_id=args.lemma_id,
        owner=m.group("user_id"),
        root=str(p.parent.parent),
        full=args.full,
        user_id=args.user_id,
        model=args.model,
        host=args.host,
        api_key=args.api_key,
        backend=args.backend,
        num_ctx=args.num_ctx,
        verbose=args.verbose,
        parser=parser,
    )
    # verify_entry returns the exit code: 0 accepted, 1 rejected (its
    # refutation written), 3 no verdict (the server failed; nothing written).
    sys.exit(rc)


def _cmd_repair(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # Resolved first: PATH is relative to the current directory or
    # absolute, and the conjecture root is found by climbing from it — a
    # bare relative name (alice_dag.json, run inside dags/) has no parents
    # to climb until it is made absolute.
    p = Path(args.path).resolve()
    if not p.is_file():
        parser.error(
            f"{args.path} is not a file; expected a refutation file in "
            f"conjectures/NAME/refutations/"
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        parser.error(f"{args.path} is not a readable refutation file: {e}")
    if not isinstance(data, dict):
        parser.error(f"{args.path} is not a refutation file (not a JSON object)")
    for field in ("user_id", "lemma_id"):
        value = data.get(field)
        if not isinstance(value, str) or not value:
            parser.error(
                f"{args.path} has no usable {field!r}; a refutation names "
                f"the lemma by its user_id and lemma_id"
            )
    rc = main_mod.repair_entry(
        lemmas_all=args.lemmas_all,
        my_lemmas=args.my_lemmas,
        tool_mode=args.tool_mode,
        lemmas_common=args.lemmas_common,
        path=args.path,
        user_id=args.user_id,
        model=args.model,
        host=args.host,
        api_key=args.api_key,
        backend=args.backend,
        num_ctx=args.num_ctx,
        verbose=args.verbose,
        parser=parser,
    )
    # 0 repaired (the lemma re-verified or re-proved, and every dependent
    # re-verified or re-proved), 1 a re-proof failed (its lemma and its
    # dependents stay suspended), 3 a verifier gave no verdict (the server
    # failed; nothing written for that lemma).
    sys.exit(rc)


def _cmd_prune(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # Forwarded resolved: prune_entry resolves its dir again, and an
    # absolute path needs no workspace root to do it.
    root = _require_conjecture_dir(
        args.dir, args.user_id, args.workspace_root, parser
    )
    rc = main_mod.prune_entry(
        dir=str(root),
        user_id=args.user_id,
        parser=parser,
    )
    sys.exit(rc)


def _cmd_status(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # Resolved first: PATH is relative to the current directory or
    # absolute, and the conjecture root is found by climbing from it — a
    # bare relative name (alice_dag.json, run inside dags/) has no parents
    # to climb until it is made absolute.
    p = Path(args.path).resolve()
    dag = _load_user_dag(args.path, parser)
    _lemma(dag, args.path, args.lemma_id, parser)
    m = _USER_DAG_NAME.fullmatch(p.name)
    if m is None:
        parser.error(
            f"{args.path} is not named <user_id>_dag.json; PATH must be a "
            f"user's DAG file (conjectures/NAME/dags/<user_id>_dag.json), "
            f"whose name supplies the lemma's user_id"
        )
    rc = main_mod.status_entry(
        path=args.path,
        lemma_id=args.lemma_id,
        owner=m.group("user_id"),
        root=str(p.parent.parent),
        user_id=args.user_id,
        parser=parser,
    )
    sys.exit(rc)


def _cmd_export(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # Resolved first: PATH is relative to the current directory or
    # absolute, and the conjecture root is found by climbing from it — a
    # bare relative name (alice_dag.json, run inside dags/) has no parents
    # to climb until it is made absolute.
    p = Path(args.path).resolve()
    out = Path(args.out)
    if out.exists() and not out.is_dir():
        parser.error(
            f"{args.out} exists and is not a directory; -o/--out names the "
            f"directory the document is written to"
        )
    # A bare NAME is a conjecture in the workspace, unless it names a file
    # here (a DAG file, run from inside dags/).
    if p.is_dir() or (
        workspace.is_bare_name(args.path) and not Path(args.path).is_file()
    ):
        if args.lemma_id is not None:
            parser.error(
                "PATH is the conjecture directory, so the complete DAG is "
                "written; lemma_id applies to a user's DAG file"
            )
        root = _require_conjecture_dir(
            args.path, args.user_id, args.workspace_root, parser
        )
        owner = None
    else:
        dag = _load_user_dag(args.path, parser)
        m = _USER_DAG_NAME.fullmatch(p.name)
        if m is None:
            parser.error(
                f"{args.path} is not named <user_id>_dag.json; PATH must be a "
                f"user's DAG file (conjectures/NAME/dags/<user_id>_dag.json), "
                f"whose name supplies the lemma's user_id"
            )
        owner = m.group("user_id")
        if args.lemma_id is not None:
            _lemma(dag, args.path, args.lemma_id, parser)
        root = p.parent.parent
    rc = main_mod.export_entry(
        path=args.path,
        lemma_id=args.lemma_id,
        owner=owner,
        out_dir=str(out),
        root=str(root),
        user_id=args.user_id,
        parser=parser,
    )
    sys.exit(rc)


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # The run's user, resolved before any subcommand (see config.py):
    # the config's user.id, and a run without one fails
    # and names the command that sets it. The config subcommand acts on
    # the config itself, and init and new only lay out files, so they run
    # without a resolved user.
    if args.command not in ("config", "init", "new"):
        try:
            args.user_id = config.ensure_user_id()
        except config.UserConfigError as e:
            print(f"proofs: {e}", file=sys.stderr)
            sys.exit(1)

    # The workspace root, found once here and handed to the commands, so
    # the transition warning is printed once. config is per-user and init
    # makes the workspace, so neither looks for one.
    if args.command not in ("config", "init"):
        args.workspace_root = _workspace_root()

    args.func(args, args.parser)


if __name__ == "__main__":
    main()
