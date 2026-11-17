"""The `proofs` command: one entry point for every proofs tool.

    proofs config KEY [VALUE]         get or set the user's config, git-style
    proofs init [DIR]                 make DIR (default: .) a proofs workspace
    proofs new NAME                   lay out a blank conjecture, conjectures/NAME/
    proofs parse DIR                  parse DIR's references/ into references.md
    proofs run DIR                    run the proof loop on DIR
    proofs verify DIR LEMMA_ID        verify one lemma [--full]
    proofs repair path_to_refutation  repair the lemma a refutation file names
    proofs prune DIR                  remove refuted lemmas [--with-dependents]
    proofs migrate [DIR]              convert a format-1 workspace to streams
    proofs status DIR LEMMA_ID        a lemma's closure: hashes, certificates
    proofs export DIR [LEMMA_ID] [--user USER_ID] -o OUT_DIR
                                    write a LaTeX document of the DAG
    proofs limits DIR                 show the code limits in effect, and their sources
    proofs tools lock DIR             write DIR's tools.lock from its flake (maintainer)
    proofs tools check DIR            check DIR's declared tools and the sandbox

A workspace is the directory proofs init marks with a .proofs file, found
by walking up from the current directory (workspace.find_root); main()
finds it once and hands it to every command that needs it, as
args.workspace_root. conjectures/NAME below is relative to that root.

DIR is a conjecture directory: a directory holding a conjecture.md, given
as a bare NAME (looked up in the workspace's conjectures/) or as a path,
relative to the current directory or absolute. LEMMA_ID is a lemma's id,
<user_id>/<run_id>/<hex>, as the logs and proofs status show it. A path works
outside a workspace; a bare NAME, and proofs new, need one.

This module validates the arguments and hands each command to its entry
point, where the command's own docstring says what it does:

  * parse and run forward to parsing.run() and startup.main() with DIR as
    --conjecture, and pass anything after DIR (--model, --mode,
    --max-iterations, --parallel, ...) through unchanged.
  * verify (loops/verify.py) runs the three verifier checks on a lemma
    that is already in the DAG: a rejection writes a refutation file
    (refutations.py), an acceptance a certificate (certificates.py), and
    the lemma's stale refutations give their justifications to the
    verifiers either way.
  * repair (loops/repair.py) is addressed to one refutation file by its
    path. It re-verifies the lemma with the refutation's justification as
    input, deleting the file when the proof is accepted; a rejected proof
    is re-proven through the prover, verifier and reviser and committed as
    a reproof under the lemma's id, and the lemmas that depend on it are
    re-verified up the chain, each re-proved if its re-verification
    rejects it.
  * prune (core/prune.py) removes every lemma whose refutation still
    counts, with its certificates and refutation files (and, with
    --with-dependents, everything that cites it), and clears scratch.
  * status (core/status.py) reports, for each lemma in the named lemma's
    dependency closure, its hash and every valid certificate with how many
    times it was issued. It writes nothing.
  * export (export/command.py, export/latex.py) writes the DAG as a LaTeX
    document in dependency order — the complete DAG, one user's lemmas
    (--user), or a lemma and everything it cites — into the directory
    -o/--out names, created if it is not there yet. -o/--out is required:
    the document is the user's artifact, not the project's data.
  * init (workspace.init) creates DIR/conjectures/, writes the .proofs
    marker and writes or updates the proofs block of DIR/.gitignore. It
    is safe to re-run, and refuses a DIR inside another workspace.
  * new creates conjectures/NAME/ under the workspace root with an empty
    references/ directory, an empty conjecture.md and comments.md, and a
    references.md holding the empty collection []. It refuses a NAME that
    is already there.

None of the commands run git: users share a conjecture's files by
committing, pushing and pulling them by hand. Every run writes streams of
its own (core/streams.py), so two users' work never lands in the same
file. The exceptions are verify and repair, whose acceptance deletes the
lemma's refutations whoever wrote them, and prune, which rewrites the
streams holding what it removes.

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
import sys
from pathlib import Path
from typing import List, Optional

from . import flags
from ..core import migrate, prune, startup, status, streams
from ..export import command as export_command
from ..loops import repair, verify
from ..references import parsing as parsing_mod
from ..tools.run_code import codetools, sandbox
from ..tools.run_code import setup as code_setup
from ..tools.run_code import limits as limits_mod
from ..workspace import config
from ..workspace import layout as workspace

# Argument errors exit 2, argparse's convention; a command's own exit
# code is what its entry returns (0 accepted, 1 rejected, and so on).



def _add_backend_flags(sp: argparse.ArgumentParser, help_for: str) -> None:
    """The model and transport flags verify and repair take, the run's
    (cli/flags.py), and --verbose. help_for names the work in the
    --verbose help ("the verification", "the re-verification")."""
    flags.add_backend_flags(sp)
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


def _positive(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{n} is not a positive count")
    return n


def _add_parallel_flag(sp: argparse.ArgumentParser) -> None:
    """--parallel N: how many of a lemma's verifiers verify and repair run
    at once. Not args.parallel, which setup_code reads as a parallel run's
    loop count."""
    sp.add_argument(
        "--parallel", type=_positive, default=1, metavar="N",
        dest="parallel_verifiers",
        help=(
            "Run up to N of a lemma's verifier checks at once (there are 3, "
            "or 4 for a proof that rests on a computation). The first "
            "rejection ends the check without waiting for the rest; an "
            "acceptance needs them all. Default: 1, one after another."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proofs",
        description=(
            "Distributed proof construction, coordinated through git. "
            "DIR is a conjecture directory: a NAME under the workspace "
            "root's conjectures/, or a path to it. LEMMA_ID is a lemma's id, "
            "<user_id>/<run_id>/<hex>. Run proofs init to make a workspace."
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
            "the .proofs marker and a .proofs-local.json holding the default "
            "code limits, and write or update the proofs block of "
            "DIR/.gitignore (lines outside the block are left alone). Safe "
            "to re-run: existing files other than the block are kept. A DIR "
            "inside another workspace is refused."
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
        "limits",
        help="Show the code limits in effect for DIR on this machine.",
        description=(
            "Show each code limit (code.timeout, code.memory, code.jobs, "
            "...) in effect for runs on DIR, and where it comes from: "
            "conjectures/NAME/limits.json, the workspace's "
            ".proofs-local.json, or the built-in default. Both files are "
            "gitignored JSON objects holding only the keys they override; "
            "--code-* flags override both for one run."
        ),
    )
    p.add_argument("dir", metavar="DIR", help=_DIR_HELP)
    p.set_defaults(func=_cmd_limits, parser=p)

    p = sub.add_parser(
        "tools",
        help="Write or check DIR's tools.lock.",
        description=(
            "lock: write DIR's tools.lock from the proofsTools output of "
            "its flake.nix, by nix eval (the conjecture maintainer runs "
            "this, and commits both). check: report whether every tool "
            "tools.lock declares is on PATH at its version, and whether "
            "the sandbox can start here."
        ),
    )
    p.add_argument("action", choices=("lock", "check"))
    p.add_argument("dir", metavar="DIR", help=_DIR_HELP)
    p.set_defaults(func=_cmd_tools, parser=p)

    p = sub.add_parser(
        "verify",
        help="Verify one lemma, or every uncertified one.",
        description=(
            "Verify the named lemma. With --full, also verify everything it "
            "cites, cited lemmas first. With --uncertified instead of a "
            "lemma, verify every suspended lemma that lacks only a "
            "certificate, from the roots up."
        ),
    )
    p.add_argument("dir", metavar="DIR", help=_DIR_HELP)
    p.add_argument("lemma_id", metavar="LEMMA_ID", nargs="?",
                   help="The lemma to verify.")
    p.add_argument(
        "--full",
        action="store_true",
        help="Also verify everything the lemma cites, cited lemmas first.",
    )
    p.add_argument(
        "--uncertified",
        action="store_true",
        help=(
            "Verify every suspended lemma whose only fault is a missing "
            "certificate, cited lemmas first, skipping any that cites a "
            "lemma that stays suspended. A rejection writes a refutation "
            "for proofs repair and the run goes on."
        ),
    )
    _add_backend_flags(p, help_for="the verification")
    _add_parallel_flag(p)
    p.add_argument(
        "--run-code",
        action="store_true",
        help=(
            "Re-execute the scripts each lemma's proof rests on, in the "
            "sandbox, after showing them and asking you to confirm: a "
            "result that differs from the recorded one writes a refutation. "
            "Needs --tools. Read other people's code before you run it."
        ),
    )
    code_setup.add_code_flags(p)
    p.set_defaults(func=_cmd_verify, parser=p)

    p = sub.add_parser(
        "repair",
        help="Repair the lemma a refutation file names, or every refuted lemma.",
        description=(
            "Repair the lemma a refutation file names. The refutation file "
            "supplies the lemma's id, so no other arguments are needed. "
            "With --all DIR, repair every lemma a refutation counts against, "
            "cited lemmas first, each at most once."
        ),
    )
    p.add_argument(
        "path",
        metavar="path_to_refutation",
        help="A file in conjectures/NAME/refutations/ (under the "
             "workspace root); with --all, the conjecture DIR.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help=(
            "Repair every refuted lemma of the conjecture DIR, cited lemmas "
            "first. A lemma is tried once per run: one rejected again keeps "
            "its refutation for the next run. One citing a lemma that stays "
            "suspended is skipped."
        ),
    )
    _add_backend_flags(p, help_for="the re-verification")
    _add_parallel_flag(p)
    flags.add_lemma_window_flags(p)
    flags.add_tool_flags(p)
    code_setup.add_code_flags(p)
    p.set_defaults(func=_cmd_repair, parser=p)

    p = sub.add_parser(
        "prune",
        help="Remove refuted lemmas from DIR, and clear its scratch.",
        description=(
            "Remove every lemma whose refutation still counts, with its "
            "certificates and refutation files; the lemmas citing one stay, "
            "suspended, unless --with-dependents is given. Streams a run on "
            "this machine is writing to are left for the next prune. Also "
            "clears the scratch no run needs."
        ),
    )
    p.add_argument("dir", metavar="DIR", help=_DIR_HELP)
    p.add_argument(
        "--with-dependents", action="store_true",
        help="Also remove every lemma that cites a removed lemma, transitively.",
    )
    p.set_defaults(func=_cmd_prune, parser=p)

    p = sub.add_parser(
        "migrate",
        help="Convert a format-1 workspace (one file per user) to streams.",
        description=(
            "Rewrite every conjecture's format-1 files — dags/<user>_dag.json, "
            "certificates/<user>.jsonl, refutations, runs/ and code/ — as "
            "streams, one run per user, and mark the workspace format 2. Lemma "
            "ids become <user>/<run>/<old id>; proofs are not touched; valid "
            "certificates and counting refutations are re-issued under the new "
            "hash. Commit first, then review and commit the result."
        ),
    )
    p.add_argument(
        "dir", metavar="DIR", nargs="?", default=None,
        help="The workspace root (default: the workspace you are in).",
    )
    p.set_defaults(func=_cmd_migrate, parser=p)

    p = sub.add_parser(
        "status",
        help="A lemma's closure: hashes, certificates, suspension.",
        description=(
            "For each lemma in the named lemma's citation closure, its hash, "
            "every valid certificate with how many times it was issued, and "
            "whether it is suspended."
        ),
    )
    p.add_argument("dir", metavar="DIR", help=_DIR_HELP)
    p.add_argument("lemma_id", metavar="LEMMA_ID", help="The lemma whose closure to list.")
    p.set_defaults(func=_cmd_status, parser=p)

    p = sub.add_parser(
        "export",
        help="Write a LaTeX document of the DAG, in dependency order.",
        description=(
            "Write the DAG as a LaTeX document, in dependency order "
            "(dependencies first): the complete DAG; with --user, the "
            "lemmas that user's runs proved; with a lemma id, that lemma "
            "and everything it cites. The document is written into the "
            "-o/--out directory (created if it is not there yet)."
        ),
    )
    p.add_argument("dir", metavar="DIR", help=_DIR_HELP)
    p.add_argument(
        "lemma_id", metavar="LEMMA_ID", nargs="?", default=None,
        help="Optional: write this lemma and everything it cites.",
    )
    p.add_argument(
        "--user", default=None, metavar="USER_ID",
        help="Write only the lemmas this user's runs proved.",
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
        f"references/.\n"
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
    startup.main(
        argv=["--conjecture", args.dir, *args.rest], prog="proofs run",
        root=args.workspace_root,
    )


def _cmd_verify(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.uncertified and (args.lemma_id or args.full):
        parser.error("--uncertified verifies every uncertified lemma: give "
                     "it no LEMMA_ID and no --full")
    if not args.uncertified and not args.lemma_id:
        parser.error("name the LEMMA_ID to verify, or give --uncertified")
    root = _require_conjecture_dir(args.dir, args.user_id, args.workspace_root, parser)
    if args.run_code and args.tools is None:
        parser.error("--run-code runs the lemma's scripts, so it needs --tools")
    # --tools: check the tools and re-launch inside the sandbox (tools/run_code/setup.py).
    code = code_setup.setup_code(args, parser, args.workspace_root,
                                 conjecture=str(root))
    common = dict(
        code=code,
        root=str(root),
        run_code=args.run_code,
        user_id=args.user_id,
        model=args.model,
        host=args.host,
        api_key=args.api_key,
        backend=args.backend,
        num_ctx=args.num_ctx,
        verbose=args.verbose,
        parser=parser,
        parallel=args.parallel_verifiers,
    )
    if args.uncertified:
        rc = verify.verify_uncertified_entry(**common)
    else:
        rc = verify.verify_entry(lemma_id=args.lemma_id, full=args.full, **common)
    # Both return the exit code: 0 accepted, 1 rejected (its refutation
    # written), 3 no verdict (the server failed; nothing written).
    sys.exit(rc)


def _cmd_repair(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.all:
        _cmd_repair_all(args, parser)
        return
    # Resolved first: the path is relative to the current directory or
    # absolute, and the conjecture root is found by climbing from it — a
    # bare relative name (run inside refutations/) has no parents to climb
    # until it is made absolute.
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
    for field in ("lemma_id",):
        value = data.get(field)
        if not isinstance(value, str) or not value:
            parser.error(
                f"{args.path} has no usable {field!r}; a refutation names "
                f"the lemma by its id"
            )
    # --tools: the re-proving prover and reviser may run code, and the
    # whole repair runs inside the sandbox (tools/run_code/setup.py).
    code = code_setup.setup_code(args, parser, args.workspace_root,
                               conjecture=str(p.parent.parent))
    rc = repair.repair_entry(
        code=code,
        parallel=args.parallel_verifiers,
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


def _cmd_repair_all(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    root = _require_conjecture_dir(args.path, args.user_id, args.workspace_root, parser)
    code = code_setup.setup_code(args, parser, args.workspace_root,
                                 conjecture=str(root))
    rc = repair.repair_all_entry(
        code=code,
        parallel=args.parallel_verifiers,
        lemmas_all=args.lemmas_all,
        my_lemmas=args.my_lemmas,
        tool_mode=args.tool_mode,
        lemmas_common=args.lemmas_common,
        root=str(root),
        user_id=args.user_id,
        model=args.model,
        host=args.host,
        api_key=args.api_key,
        backend=args.backend,
        num_ctx=args.num_ctx,
        verbose=args.verbose,
        parser=parser,
    )
    # 0 nothing failed, 1 a lemma could not be re-proved or was refused,
    # 3 a verifier gave no verdict (the run stopped there), 130 Ctrl-C.
    sys.exit(rc)


def _cmd_prune(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # Forwarded resolved: prune_entry resolves its dir again, and an
    # absolute path needs no workspace root to do it.
    root = _require_conjecture_dir(
        args.dir, args.user_id, args.workspace_root, parser
    )
    rc = prune.prune_entry(
        dir=str(root),
        user_id=args.user_id,
        parser=parser,
        with_dependents=args.with_dependents,
    )
    sys.exit(rc)


def _cmd_migrate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    sys.exit(migrate.migrate_entry(dir=args.dir, parser=parser))


def _cmd_limits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    root = _require_conjecture_dir(
        args.dir, args.user_id, args.workspace_root, parser
    )
    try:
        lim = limits_mod.resolve(workspace.find_root(root), root)
    except limits_mod.LimitsError as e:
        parser.error(str(e))
    print(lim.describe())
    errors, warnings = limits_mod.check(lim)
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"error: {e}")
    sys.exit(1 if errors else 0)


def _cmd_tools(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    root = _require_conjecture_dir(
        args.dir, args.user_id, args.workspace_root, parser
    )
    if args.action == "lock":
        try:
            written = codetools.write_lock(root)
        except codetools.ToolError as e:
            parser.error(str(e))
        print(f"wrote {written}; commit it with the flake")
        sys.exit(0)
    ok = True
    try:
        lock = codetools.load_lock(root)
        names = codetools.requested("all", lock, root)
    except codetools.ToolError as e:
        parser.error(str(e))
    for name in names:
        try:
            f = codetools.find(name, lock, root)
            print(f"ok    {name} {f.version}  ({f.path})")
        except codetools.ToolError as e:
            ok = False
            print(f"FAIL  {name}: {e}")
    try:
        b = sandbox.check()
        print(f"ok    sandbox: {b.name} ({b.exe})"
              + ("" if b.outer else "; scripts only — proofs itself is not confined here"))
    except sandbox.SandboxError as e:
        ok = False
        print(f"FAIL  sandbox: {e}")
    scope = sandbox.systemd_scope()
    print(f"{'ok  ' if scope else 'note'}  memory backstop: "
          f"{'systemd-run user scope' if scope else 'none (watchdog only)'}")
    sys.exit(0 if ok else 1)


def _cmd_status(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    root = _require_conjecture_dir(args.dir, args.user_id, args.workspace_root, parser)
    rc = status.status_entry(
        lemma_id=args.lemma_id,
        root=str(root),
        user_id=args.user_id,
        parser=parser,
    )
    sys.exit(rc)


def _cmd_export(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    out = Path(args.out)
    if out.exists() and not out.is_dir():
        parser.error(
            f"{args.out} exists and is not a directory; -o/--out names the "
            f"directory the document is written to"
        )
    if args.lemma_id is not None and args.user is not None:
        parser.error("give a LEMMA_ID or --user, not both")
    root = _require_conjecture_dir(args.dir, args.user_id, args.workspace_root, parser)
    rc = export_command.export_entry(
        lemma_id=args.lemma_id,
        owner=args.user,
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
    if args.command not in ("config", "init", "new", "migrate"):
        try:
            args.user_id = config.ensure_user_id()
        except config.UserConfigError as e:
            print(f"proofs: {e}", file=sys.stderr)
            sys.exit(1)

    # The workspace root, found once here and handed to the commands, so
    # the transition warning is printed once. config is per-user and init
    # makes the workspace, so neither looks for one.
    if args.command not in ("config", "init", "migrate"):
        args.workspace_root = _workspace_root()

    try:
        args.func(args, args.parser)
    except streams.StreamError as e:
        # A stream that cannot be read — a git conflict left in it, a line
        # that is not a record — stops every command, named by file and line.
        args.parser.error(str(e))


if __name__ == "__main__":
    main()
