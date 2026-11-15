# Multi-Agent Theorem Prover

A Python program that drives a small team of LLM agents to iteratively prove
(or refute) a mathematical conjecture. The agents work in a loop —
**planner → prover → verifier → reviser** — and accumulate a proof
as a DAG of lemmas. Which planned lemma is proved next is not an agent's
call: in `--mode auto` it is picked at random (serial) or taken oldest-first
from a queue (parallel); in `--mode human` the operator picks at the menu.

The system is distributed and coordinated through git. Several people can
work on one conjecture: every run appends its lemmas to an append-only stream
of its own, `conjectures/NAME/dags/<user_id>/<run_id>/000001.jsonl`, and
every command reads all the runs' streams as one complete DAG. Every acceptance is recorded as a
*certificate* and every rejection on re-verification as a *refutation*,
both tied to a hash of what the verifiers were shown: the lemma and the
statements it cites. A lemma that has no valid certificate, or has a
refutation against it, or cites a lemma in either state, is *suspended* and
is not built on until it is verified again. See
[Sharing a conjecture](#sharing-a-conjecture).

By default the loop is serial: one lemma in flight at a time. `--parallel N`
runs N of these loops side by side around one shared DAG, steered by a planner
and a lemma generator that work for all the loops at once — see
[Parallel mode](#parallel-mode).

## How it works

Each iteration:

1. **Planner** (`planner.md`) reads the conjecture, the lemmas proved so far,
   and the most recently rejected lemmas, and proposes
   `PLANNER_CANDIDATES` (5) candidate lemmas, in no particular order, working either
   toward a proof of the conjecture or — where it looks false — toward a
   concrete counterexample to it. The code gives each candidate its
   permanent id as it is proposed (see [Sharing a conjecture](#sharing-a-conjecture)),
   so no two candidates can collide and none is screened out.
2. **Selection** picks exactly one candidate:
   - `--mode auto`: one candidate is picked uniformly at random.
   - `--mode human`: a menu appears. Type a number and Enter to prove that
     lemma yourself: paste or type your proof and its citations, and it
     goes through the same three verifiers as the prover's (see
     [Proving a lemma yourself](#proving-a-lemma-yourself)). Type `Np` to
     send a lemma through the ordinary proving pipeline instead. You can
     also write your own lemma (`w` to prove it yourself, `wp` to have it
     proved) or send the planner back. Nothing enters the DAG without
     passing the verifiers.
3. **Prover** (`prover.md`) writes a complete, rigorous proof of the chosen
   lemma, citing only lemmas already in the DAG.
4. **Verifiers** (`verifier_1.md`, `verifier_2.md`, `verifier_3.md`) check it
   adversarially — logic *and* arithmetic. Verification is three atomic
   steps, one per agent; a lemma enters your run's stream only after all
   three have accepted it, and the acceptance is recorded as a certificate in
   your run's certificate stream. The three prompts are specialized: `verifier_1.md` checks
   logic and computations, `verifier_2.md` specializes in logic (definitions,
   cited results, the conclusion), and `verifier_3.md` in computations and
   completeness.
5. **Reviser** (`reviser.md`) handles a reject, judging the lemma's
   difficulty from the rejected proof itself (the *last proof*), which it
   sees alongside the lemmas already proved and the known references. It
   decides
   whether the fault is in the *proof* (keep the statement; send the prover
   back with the verdict as feedback), in the *statement* (return a revised
   statement as the prover's new target), or whether the lemma is *too hard*
   to prove as stated (decompose it into a smaller lemma — a sub-statement
   the last proof left unjustified, under a new id the code draws — and start
   the prover on that with a fresh budget). After `MAX_PROOF_ATTEMPTS` (default 3) prover
   rounds the lemma is given up and the planner is asked again. When the
   prover instead overflows the context window, the same reviser does the
   same job from the partial proof — see
   [When a call hits the context wall](#when-a-call-hits-the-context-wall).

### When the conjecture is settled

No agent can declare the conjecture proved. It is settled by a lemma, and a
candidate attempts it by naming which way in its `settles` field:

- `conjecture`: the conjecture itself.
- `conjecture_negation`: that the conjecture is false, by an explicit
  counterexample.

The planner (or, in parallel mode, the lemma generator) attempts the
conjecture by proposing a candidate with `settles` set; at the human menu,
write a lemma whose statement is just `conjecture` or
`conjecture_negation`. Its statement
is always written by the code, not by the model: the verbatim text of
`conjecture.md`, or that text after the fixed prefix "The following
conjecture is false. Exhibit an explicit counterexample…". From there it is
an ordinary candidate. It is picked like any other, with no priority, the prover assembles its proof from the DAG, and the three
verifiers check it. The reviser may not revise a pinned statement.

Whether the conjecture is settled is computed, never stored
(`proofs/core/resolution.py`). It is settled while some lemma, anyone's, meets
both conditions:

- Its statement is exactly the pinned text, up to whitespace.
- It is not suspended, so it holds a valid certificate and everything it
  cites, all the way down, is certified and unrefuted.

The test is on the statement, so a proof committed under a renamed id
(`conjecture_2`) counts too. The serial run checks at the top of every
iteration and once after the last; the parallel planner checks on every DAG
change. Either way, a run stops when a DAG it started from — another
user's pulled file among them — settles the conjecture. Editing `conjecture.md`, or a refutation or repair anywhere
below the settling lemma, reopens the conjecture on its own. A DAG that
proves both the conjecture and its negation
settles nothing: the run warns and carries on.

### Thinking vs. structured output

The prompts instruct each model to emit strict JSON directly, so every
response is first parsed as-is; a schema-constrained *extraction* call only
runs as a fallback. The underlying tension there — on llama.cpp, reasoning
silently voids the grammar — is handled inside `proofs/llm/openai_compat.py`, which
disables reasoning for exactly the calls that carry a schema.

### When a call hits the context wall

The server is stateless, so a response cut off by the context window cannot
just be re-sent. Before reporting a ceiling, `reason()` runs a pi-style
compaction rescue: the thinking trace (scratch) is summarised into a
structured summary — in chunked passes if it is big — the answer written so
far is kept verbatim, and the model is asked to resume from the cut with the
whole headroom of the now-smaller prompt. A continuation that hits the wall
itself is not a failure but the next pass: its trace is folded into the
summary, its text extends the verbatim answer, and the model resumes from
the new cut.

The rescue is bounded to two compaction-and-resume passes
(`MAX_COMPACTION_PASSES`): the first usually finishes an answer that merely
ran long, the second covers the genuinely long proofs. Still unfinished then
— or when a pass leaves nothing to build on, or no headroom is left —
`reason()` reports the ceiling and hands back the partial work. The proof
loop compacts it a final time (folding the last cut-off trace into the
summary) and sends the summary plus the verbatim partial answer to the
reviser, which now reads an overflow instead of a rejected proof and picks
a smaller lemma the partial work was building toward — or, if the partial
work shows the statement itself is false or badly posed, returns a
corrected statement. If it finds a smaller lemma, the overflowing lemma is
set aside — its history recorded under its own id — and the prover starts
on the smaller lemma with a fresh budget; a corrected statement keeps the
lemma's id and the prover re-runs on it. Only when there is nothing left
to build on, or the reviser has neither a usable smaller lemma nor a usable
revision, does the lemma fall back to the planner's decomposition escape
hatches.

All of that assumes the server *answers* the call and merely cuts it off.
When the window the run budgets (`NUM_CTX`) is larger than the window the
server actually has — a `--host` API, where nothing is probed, or a
`--num-ctx` above the server's own limit — the server does not truncate:
OpenAI- and vLLM-style servers answer a call whose prompt plus requested
generation outruns their window with a 400 that names the real ceiling
("maximum context length is X tokens"). `chat()` reads that ceiling out of
the 400 and raises `ContextLengthError` rather than a transport error, so
no retry loop burns its attempts on a 400 that is guaranteed to repeat, and
the run stops with the fix spelled out: `--num-ctx X`, matched to the
server's `--max-model-len` — or, when the prompt alone already exceeds the
window, the fact that no `--num-ctx` fixes that and the DAG has grown past
the model. The checkpoint is untouched, so the corrected re-run resumes
from the last safe boundary.

### Backend

One transport, probed at startup (`proofs/llm/openai_compat.py`, behind the
small `Backend` interface in `proofs/llm/base.py` that the loops call): the
OpenAI-compatible `/v1/chat/completions` endpoint. The same code serves two
backends:

- **llama.cpp** — a local `llama-server`. There the grammar suppresses
  thinking, so reasoning is disabled for schema calls.
- **a hosted API** — any OpenAI-compatible provider (OpenAI, OpenRouter,
  DeepSeek, ...), pointed at by `--host` with its base URL and authenticated
  with `--api-key`. A supplied key is checked against the
  provider's `/v1/models` at startup, so a wrong key fails loudly instead of
  401-ing every call in a run.

The probe can't tell the two apart by name, so it tries: a `/props` answer
means llama.cpp (full option set, real context ceiling); no `/props` means a
generic endpoint (standard OpenAI option set — `top_k`/`min_p` are dropped,
because a strict provider 400s on them — and the context limit is a default
unless `--num-ctx` is given).

`--backend {auto,llamacpp,openai}` (default `auto`) overrides that decision
when the probe can't see the real server — a proxy in front of llama-server
that eats `/props`, or a provider that happens to serve one. An explicit
backend is taken on faith, and a startup warning says when it contradicts
the probe (a forced llama.cpp against a server with no `/props` route, or a
forced standard OpenAI against one that has it). The probe's *measurements* — the
context ceiling and the loaded model — are kept either way, because they
describe the server that is actually there.

**What `--model` does, and which model gets recorded.** `--model` is sent
as the `model` field of every request, and that is all it does. A hosted
API serves the model it names. A llama-server ignores it: it answers with
whatever model it loaded at launch, and names its `--alias` in the reply.

Certificates and refutations therefore record the model that actually did
the work, not the `--model` request:

- **On a llama-server**, the file name of the loaded model, read off
  `/props` at startup (e.g. `Qwen3.8-27B-UD-Q3_K_XL.gguf`).
- **Anywhere else**, the model named in the reply's own `model` field — for
  a hosted API, the model `--model` selected, usually with its version
  suffix (`gpt-4o` → `gpt-4o-2024-08-06`).
- **When the server names nothing**, the `--model` name.

The `/v1/models` list is not used for naming: on a hosted API it is the
provider's whole catalogue.

Model capabilities (context size) are *profiles*, probed
rather than hard-coded, so adding a model needs no code change.

## Setup

A Nix dev shell is provided:

```sh
nix develop
```

It gives you a Python 3 environment with `requests` installed and puts the
`proofs` command on your PATH. The command is also exposed as a flake app,
so `nix run . -- --help` works without cloning. Outside Nix, `pip install .`
installs the `proofs` command; otherwise any Python 3 install with
`requests` works, with this repository on `PYTHONPATH`:
`PYTHONPATH=/path/to/Proofs python -m proofs ...`, run from inside your
workspace. If [`orjson`](https://github.com/ijl/orjson) is installed, the
DAG is read with it — it reads the streams about twice as fast, and reading
them is most of a large DAG's load — and with the standard library otherwise;
both read the same records the same way.

The program and your conjectures live apart: this repository holds only
the code and the default prompts, and a workspace — anywhere on disk,
usually its own git repository — holds the conjectures. Every command
finds its workspace from the directory it is run in, so the same install
serves every workspace you have.

Then make a workspace — the directory your conjectures live in — set your
identity once, and start a conjecture:

```sh
proofs init ~/my-proofs        # or `proofs init` in the current directory
cd ~/my-proofs
git init                       # if it is not already in a repository
proofs config --global user.id eoin
proofs new standard_d          # then write the statement into conjectures/standard_d/conjecture.md
proofs run standard_d
```

See [Workspaces](#workspaces).

## Usage

### The `proofs` command

One command, twelve subcommands:

| Subcommand | Takes | Does |
| --- | --- | --- |
| `proofs config KEY [VALUE]` | a config key, an optional value | Gets or sets the user's config, git-style: `user.id`, `user.name`, `user.email`. |
| `proofs init [DIR]` | a directory (default `.`) | Makes `DIR` a proofs workspace: creates `DIR/conjectures/`, writes the `.proofs` marker (`format = 2`) and a `.proofs-local.json` holding the default code limits, and writes or updates the proofs block of `DIR/.gitignore` (between `# >>> proofs >>>` and `# <<< proofs <<<`; lines outside it are never touched). Safe to re-run; refuses a `DIR` inside another workspace. |
| `proofs new NAME` | a directory name | Lays out a blank conjecture, `conjectures/NAME/` under the workspace root: an empty `references/` directory, an empty `conjecture.md` and `comments.md`, and a `references.md` holding the empty collection `[]`. Refuses a `NAME` that already exists. |
| `proofs run DIR` | `DIR`, then the loop's options | Runs the proof loop on the conjecture. |
| `proofs parse DIR` | `DIR`, then the parser's options | The maintainer's tool: parses `DIR`'s `references/` into the committed `references.md`. |
| `proofs verify DIR LEMMA_ID [--full]` | a conjecture, a lemma id | Verifies the lemma; with `--full`, what it cites too, cited lemmas first. With `--tools ... --run-code`, re-executes the lemma's [computations](#code-tools-sage-python) first, after asking. |
| `proofs repair path_to_refutation` | a refutation file | Repairs the lemma the refutation names. |
| `proofs prune DIR [--with-dependents]` | a conjecture directory | Removes every lemma whose refutation still counts, with its certificates and refutation files (and, with `--with-dependents`, everything citing it); compacts repaired lemmas to their current version and drops certificates that match no current hash; and clears scratch no run needs. See [Pruning](#pruning). |
| `proofs migrate [DIR]` | a workspace root (default: the one you are in) | Converts a format-1 workspace (one file per user and kind) to streams. See [Migrating a format-1 workspace](#migrating-a-format-1-workspace). |
| `proofs status DIR LEMMA_ID` | a conjecture, a lemma id | Lists the lemma's citation closure: each lemma's hash, stream file, valid certificates with how many times each was issued, and suspension; then whether the conjecture is settled. |
| `proofs limits DIR` | a conjecture directory | Prints the [code limits](#limits) in effect on this machine and where each comes from. |
| `proofs tools lock\|check DIR` | a conjecture directory | `lock`: the maintainer's tool — writes `tools.lock` from the conjecture's `flake.nix`. `check`: whether each declared tool is on PATH at its version, and whether the sandbox starts. See [Code tools](#code-tools-sage-python). |
| `proofs export DIR [LEMMA_ID] [--user USER_ID] -o OUT_DIR` | a conjecture, an optional lemma id or user, the output directory | Writes a LaTeX document of the DAG, in dependency order, into OUT_DIR. |

Every command works inside a **workspace**, the directory `proofs init`
marks with a `.proofs` file. A bare `NAME` always means `conjectures/NAME`
under the workspace root, wherever in the workspace the command is run; a
path is taken relative to the current directory, as usual. See
[Workspaces](#workspaces).

`DIR` is a conjecture directory: a bare `NAME`, looked up in the
workspace's `conjectures/`, or a path to one. `LEMMA_ID` is a lemma's id,
`<user_id>/<run_id>/<6 hex digits>` — the user and the run that proposed it,
and a random part — as the run's log and `proofs status` show it. `export` writes the DAG's LaTeX document — the
complete DAG, the lemmas one user's runs proved (`--user`), or a lemma and
everything it cites — in dependency order (dependencies first), into the
directory `-o`/`--out` names (created if it is not there yet — the document is
the user's artifact, and never lands in the project on its own): `proof.tex`,
`<user_id>_proof.tex`, or the lemma's id with its slashes as `--`, plus
`_proof.tex`.

Every run acts as a **user**, named by a `user.id` that names its streams,
begins the ids of the lemmas it commits, and is recorded in its certificates
and refutations. The user's identity —
`user.id`, `user.name` (a full name) and `user.email` — is stored in
`~/.config/proofs/config` (or `$XDG_CONFIG_HOME/proofs/config`), outside the
repository, so it is never committed. It is set and read git-style, through
the `config` subcommand: `proofs config --global user.id ID` sets (likewise
`user.name` and `user.email`), `proofs config user.id` prints, an empty
value unsets, and `proofs config --list` lists the set keys. A run without
a `user.id` fails and names the command to set it; proofs does not ask for
one, as git does not.

The identity is also stamped in the header line every stream file starts
with — `user.id`, `user.name`, `user.email` and the run id — so a file read
out of context still says whose it is, and who wrote what. The model is not
in the header, since a resumed run may use another: each lemma, certificate
and refutation names the model that produced it. Every run writes only streams of its own, so two
people who happen to pick the same `user.id` never write the same file; the
stamp tells their runs apart.

```sh
# once per workspace: conjectures/, the .proofs marker, the .gitignore block
proofs init

# a bare name is looked up in the workspace's conjectures/, from anywhere
# inside the workspace; a path (conjectures/algebra_example, /abs/path)
# works too, and outside a workspace only a path does
# --host and --model are required on every run, parse, verify and repair
proofs run algebra_example --host http://localhost:8081 --model qwen3.8-27b
proofs run algebra_example --host http://localhost:8081 --model qwen3.8-27b --mode human
proofs run algebra_example --host http://localhost:8081 --model qwen3.8-27b --parallel 2
proofs run conjectures/algebra_example --host http://localhost:8081 --model qwen3.8-27b

# or against a hosted API instead of a local llama-server
proofs run algebra_example \
    --host https://api.openai.com --model gpt-4o --api-key sk-...

# once, before proving: parse the conjecture's reference collection
proofs parse algebra_example

# the DAG as a LaTeX document, in dependency order: the complete DAG, or
# one user's lemmas, or a single lemma with everything it cites, written
# into the directory -o names (created if it is not there yet)
proofs export algebra_example -o ~/tex
proofs export algebra_example --user alice -o ~/tex
proofs export algebra_example alice/3f9a2c1b4d5e6f70/a41c07 -o ~/tex

# start a new conjecture (conjectures/standard_d/ under the workspace
# root), then write its statement into conjecture.md
proofs new standard_d

# on a fresh machine, set the identity once, git-style
proofs config --global user.id eoin
proofs config --global user.name "Eoin Mackall"
proofs config --global user.email eoin@example.com
```

`run` and `parse` both take one argument, the conjecture directory: a name
under the workspace root's `conjectures/` or a path to it — anything after it is passed through
to the tool unchanged. `parse` shares `run`'s server options (`--host`,
`--api-key`, `--backend`, `--model`, `--num-ctx`) and its DIR / `--verbose`
pair; it has no proving options, because it has no loop. `verify` and
`repair` take the same server options too, and `repair` also takes the
lemma-window and tool options (`--lemmas-all`, `--my-lemmas`,
`--lemmas-common`, `--tool-mode`), since its re-proof runs the prover and
reviser. `prune`, `status` and `export` talk to no model and take no
server options.

### CLI options

| Option | Default | Meaning |
| --- | --- | --- |
| `DIR` | — | The conjecture directory: a name under the workspace root's `conjectures/` holding a `conjecture.md`, or a path to it. Taken by `run` and `parse`. |
| `--model NAME` | required | Sent as the `model` field of every request. On a hosted API, the model to use (e.g. `gpt-4o`); a llama-server ignores it and serves the model it loaded. Not what certificates record — see [Backend](#backend). |
| `--host URL` | required | OpenAI-compatible base URL — a local llama-server, or a hosted API's base (e.g. `https://api.openai.com`, without the `/v1` suffix — it is added automatically). |
| `--api-key KEY` | none | Sent as `Authorization: Bearer <key>`. Needed for a hosted API; leave unset for a plain local llama-server. |
| `--backend {auto,llamacpp,openai}` | `auto` | Which dialect of the OpenAI-compatible endpoint to speak: the full llama.cpp option set (with thinking suppressed for schema calls), or the standard OpenAI set. `auto` takes the probe's answer; the other two force it, for when the probe can't see the real server. |
| `--max-iterations N` | `run.max_iterations`, -1 | Loop iterations before the run stops (per loop in a parallel run). `-1` runs until the conjecture is settled or you interrupt it. See [Settings](#settings). |
| `--max-proof-attempts N` | `run.max_proof_attempts`, 3 | Prover rounds per lemma before it is given up. |
| `--max-rejected-lemmas N` | `run.max_rejected_lemmas`, 4 | How many of the most recently rejected lemmas — ones that went through every prover round without being proved — the planning agents are shown, by statement only. `0` shows none. |
| `--lemmas-all N` | 25 | How many of the most recently proved lemmas (anyone's) the agents see in full; every other proved lemma is listed by id only. `-1` shows all. Also taken by `proofs repair`. |
| `--my-lemmas N` | 25 | How many of your own most recently proved lemmas the agents also see in full, overlap with `--lemmas-all` shown once. `-1` shows all. Also taken by `proofs repair`. |
| `--lemmas-common K` | 10 | How many *common lemmas* the agents also see, statement only: the lemmas outside the two recency windows that the shown lemmas cite most often (direct citations only). Ties are broken in a random order fixed for the run. A lemma nothing shown cites never ranks. `-1` shows every cited lemma. Also taken by `proofs repair`. |
| `--tool-mode {auto,json}` | `auto` | How agents call tools such as `lookup_lemmas`. `auto` uses the server's native tool calling and falls back to a JSON protocol the first time the server refuses it (llama-server needs `--jinja` for native calls); `json` uses the protocol from the start. Also taken by `proofs repair`. |
| `--tools LIST` | off | Let the prover and the reviser run code: a comma-separated list of tools the conjecture's `tools.lock` declares (`sage`, `python`), or `all`. The whole run is then sandboxed. See [Code tools](#code-tools-sage-python); with `--parallel`, see [Coding loops](#coding-loops). |
| `--code-timeout`, `--code-memory`, `--code-jobs`, `--code-threads`, `--code-max-tasks`, `--code-output-cap`, `--code-reserve-memory` | see `proofs limits` | Override one [code limit](#limits) for this run. |
| `--code-overcommit` | off | Run even when `code.jobs` × `code.memory` + `code.reserve_memory` exceeds this machine's memory. |
| `--unsandboxed-code` | off | Run code without the sandbox (where bubblewrap is unavailable). Scripts then run with your permissions, network and files. |
| `--fresh` | off | Start a new run, with an empty checkpoint, instead of continuing your most recent one. The DAG is kept. See [Cancelling a run](#cancelling-a-run-ctrl-c-and-resuming). |
| `--resume RUN_ID` | off | Continue the given run from its checkpoint. |
| `--parallel [N]` | off | Run N proof loops in parallel around one shared DAG, steered by one planner and one lemma generator. Bare `--parallel` is N=1. `--mode human`, when set, takes the place of one of the loops (loop-0). See [Parallel mode](#parallel-mode). |
| `--num-ctx N` | the server's own (probed) | Context window in tokens. On a local llama-server: defaults to the `-c` it was launched with, as probed — lower it if VRAM is tight — and an explicit value is clamped to that ceiling. On a hosted `--host` API nothing is probed, so the default is 65536 and an explicit value replaces it. Budgeting above the server's own window is caught at once from the server's 400, which names its real limit and the `--num-ctx` to re-run with. |
| `--mode {auto,human}` | `auto` | Automated selection vs. human menu at the planning step. |
| `--hotkey KEY` | `h` | Keystroke that toggles auto/human mid-run (pass `''` to disable). |
| `--verbose / --no-verbose` | on | Console logging. |
| `--ui {auto,live,plain}` | `auto` | `live` keeps a [status board](#the-status-board) of every agent at the bottom of the terminal, under the scrolling log; `plain` prints the log lines only. `auto` is live on a terminal (with `--verbose` and `rich` installed), plain otherwise. |

### The lemma window and tools

Agents are not shown every proved lemma. They see in full the
`--lemmas-all` most recently proved lemmas (anyone's) and the `--my-lemmas`
most recently proved of your own, chosen after suspended lemmas are removed
and with the overlap shown once; recency is the `proved_at` stamp a lemma
gets when it is committed or re-proved. They also see, statement only, up
to `--lemmas-common` older lemmas that the shown lemmas cite most often,
counting direct citations, with ties broken in a random order that is fixed
for the run (so prompts do not change from call to call for no reason). The
prompt therefore holds a bounded number of statements however large the DAG
grows; only the ID index grows with it. Every other proved lemma is listed
by id only, and the agent can read any of them with
the `lookup_lemmas` tool (statement and citations, never a suspended
lemma). When the window holds every lemma, no index and no tool are sent
and the prompts carry neither.

Each tool is a plug-in in its own package under `proofs/tools/`
(`lookup_lemmas/`, `run_code/`), listed in a registry in
`proofs/tools/__init__.py`; for every agent call each one decides whether
to offer itself. A tool is a name, a JSON Schema for its arguments, a
handler, a hint (what the prompt tells the model about it) and its own
round limit (`proofs/tools/base.py`). `reason()` runs the loop — at most
`tools.max_rounds` (3) rounds per call unless a tool asks for more,
`tools.max_lookup` (10) lemmas per lookup (see [Settings](#settings)) —
using native tool calls where the server supports them and a JSON protocol
otherwise (`--tool-mode`). The tools' hints are added to the
task only while the tools are offered: once they are closed (rounds spent,
no room left, or a compaction rescue) the task is re-sent without them,
with the tool results and a line saying the tools are closed.

### Code tools (Sage, Python)

With `--tools`, the prover and the reviser may call `run_code`: they write
one complete script, it runs in a fresh sandboxed process, and they read
back its status, its `RESULT` line and the end of its output. There is no
REPL: nothing carries over from one call to the next, so the script on
disk is exactly what ran. Every script ends by printing exactly one line
`RESULT: <json>`, and puts what it checks in `assert`s, so exit status 0
means the claim held on what it examined (`agents/coding.md` is the full
contract the agents are given — a shared part and one section per role;
override it per conjecture like any prompt). The verifiers get the tool
too, under one rule: a computation may justify a rejection (a
counterexample to a step), never an acceptance.

**Proofs that rest on a computation.** Only an exact, decidable claim may
rest on code. The proof stays prose: it says what the script checks and
why that settles the step, and refers to the script as `[code:<id>]`, the
id `run_code` reported. The prover lists the runs it relies on in one more
field of its answer, `code_runs`, by `run_id`; the loop reads each run
back from the scratch directory, so the recorded output is the harness's,
never the model's, and sends the proof back before any verifier sees it
when a run is unknown, failed, or a `[code:<id>]` names no cited script.
The lemma then stores the computations it cites:

```json
"code": [{"hash": "<sha256 of the script>", "lang": "sage", "produced_output": {"checked": 1000}}]
```

and, when it is committed, the scripts are copied into
`conjectures/NAME/code/<user_id>/<run_id>/<sha256>.sage` and the executions
that produced them recorded in the run's stream
`conjectures/NAME/computations/<user_id>/<run_id>/` (both committed) before
the lemma is written — so who ran a script, in which run and under which
model, is always on record. An execution record is one execution — script,
tool name, version and path, result — and how many times a script was run
is counted when asked. The Merkle hash covers the `code` field: editing a
script, or the result recorded for it, makes the lemma a different proof,
uncertified.

Every verifier sees each cited script and its recorded output, and a
fourth step, `coding_verifier.md`, runs for a proof with computations: it
checks the reduction to the finite check, the script's bounds against it,
that the claim is asserted rather than printed, exact arithmetic, no
`proof=False` or Monte Carlo methods. Nothing is run again for the
verifiers; a script whose bytes on disk no longer match its hash is shown
to them as missing. `proofs verify` works the same way on a stored lemma,
with or without `--tools`.

**Running the scripts again.** `proofs verify DIR LEMMA_ID --tools sage
--run-code` re-executes the scripts a lemma rests on before its verifiers
read it. It prints every script in full and asks before running anything
— the scripts are other people's code, and the sandbox is a second line of
defence, not the first: read them. Each script must exit 0 and print the
recorded `RESULT`:

* every script reproduces: each execution is recorded in the verifying
  run's computations stream, and the verifiers read the proof as usual;
* one does not (a failed assert, a different `RESULT`): a refutation is
  written, model `exec:<tool> <version>`, its justification saying what the
  script did and what the lemma records — the lemma is suspended until the
  computation is repaired;
* a run hits this machine's limits, a script needs a tool the run was not
  given, or you decline: inconclusive, nothing is recorded either way, and
  the verifiers read the proof as usual.

A refutation from a re-execution can only be answered by a re-execution:
while one counts, `proofs verify` without `--run-code` gives no verdict
rather than let a reading dismiss it, and `proofs repair` on one needs
`--tools` and re-runs the scripts first (re-proving the lemma if they still
do not reproduce). `proofs repair --tools` also lets the re-proving prover
and reviser run code, sandboxed like a run.

`proofs status` lists each lemma's computations: its recorded `RESULT`, and
how many recorded executions reproduced it, with which tools. `proofs
export` adds a Computations
appendix: each script the exported lemmas rest on, verbatim, with its
recorded `RESULT`, its SHA-256 and the tools recorded running it.

**Which tools, at which versions.** The conjecture's maintainer owns a
`flake.nix` beside `conjecture.md` that provides the tools and exports
their versions as `proofsTools`:

```nix
{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python3.withPackages (ps: [ ps.sympy ]);
    in {
      devShells.${system}.default = pkgs.mkShell { packages = [ pkgs.sage python ]; };
      proofsTools = {
        tools = {
          sage.version = pkgs.sage.version;
          python.version = pkgs.python3.version;
          python.packages.sympy = pkgs.python3Packages.sympy.version;
        };
        policy.result_bytes = 65536;   # the most a RESULT line may hold
      };
    };
}
```

`proofs tools lock NAME` writes that object to `tools.lock` with `nix eval`
(which evaluates the flake and runs none of it), and the maintainer commits
both. Everyone else reads only `tools.lock`. Review a flake you pulled
before you `nix develop` it: its `shellHook` runs on your machine, outside
any sandbox. `--tools sage` then needs `sage` on your PATH at the declared
version — from `nix develop conjectures/NAME`, or installed any other way —
and fails naming the problem otherwise. `proofs tools check NAME` reports
on every declared tool and on the sandbox.

**The sandbox.** A run given `--tools` re-launches itself inside
bubblewrap (Linux), in two layers:

* *proofs itself* keeps the network (it must reach the LLM server) but
  sees the filesystem read-only, `$HOME` hidden except its config and its
  own code, and may write only inside the workspace;
* *each script* has no network, an environment built from nothing (no API
  key reaches it), the system's software directories read-only, nothing
  of `$HOME` or the workspace, and one writable work directory. It runs in
  its own session and PID namespace, so killing it kills everything it
  started.

Ctrl-C works as without tools: running scripts are killed, the checkpoint
is written, and re-running the command resumes.

On **macOS** each script runs under `sandbox-exec` instead, with a profile
generated per execution: no network, writes only in its work directory,
`$HOME` unreadable apart from that directory and the tool's own install
prefix. There is no outer layer there — macOS cannot apply a second
sandbox inside a sandboxed process — so proofs itself runs unconfined, and
the scripts, the untrusted part, are confined one by one. Without a PID
namespace a script that leaves its process group (`setsid`) escapes the
kill, and memory is policed as resident size. The macOS backend has not
yet been tested on a Mac. Where neither backend can run,
`--unsandboxed-code` runs scripts with your own permissions — only for code
you would run yourself.

Every execution is kept, gitignored, in
`conjectures/NAME/scratch/<user_id>/<run_id>/`: the script, its stdout and
stderr, and `run.json` (tool version and path, platform, limits, wall
time, peak memory, exit status, `RESULT`, output hashes).

#### Limits

Limits describe the machine, not the conjecture, so they live in
gitignored files, each a JSON object holding only the keys it overrides:
`<workspace>/.proofs-local.json` for the machine (`proofs init` writes it
with every key at its default), and
`conjectures/NAME/limits.json` for one conjecture on it, which wins key by
key. A `--code-*` flag beats both for one run; `proofs limits NAME` prints
what is in effect and where each value came from.

| Key | Default | Meaning |
| --- | --- | --- |
| `code.timeout` | 600 | Seconds one execution may run. |
| `code.memory` | 8G | Memory one execution may hold, all its processes together. |
| `code.jobs` | 1 | Code job slots on this machine, shared by every proofs command on it: a serial run, `verify` or `repair` with `--tools` takes one, a parallel run one per [coding loop](#coding-loops). |
| `code.threads` | 1 | Threads one execution is told to use (OpenMP, BLAS, Sage); as many as you pass. |
| `code.max_tasks` | 32 | Processes and threads one execution may have at once: a guard against a runaway fork. It counts threads, so keep it above `code.threads`. |
| `code.output_cap` | 1M | Bytes of output one execution may write. |
| `code.reserve_memory` | 4G | Memory kept back for proofs itself — raise it when a local llama-server shares the machine. |

An execution past a limit is killed and the agent is told which limit, so
it can make the computation smaller. A run is refused when `code.jobs` ×
`code.memory` + `code.reserve_memory` exceeds physical memory (unless
`--code-overcommit`), since that is what takes a machine down; more threads
than the machine has only slows things down, so that is a warning. Where
`systemd-run` is available the whole sandboxed run also sits in one
systemd scope capped at that total, so a runaway the watchdog misses is
killed by the kernel instead of the machine running out.

### Modes and the hotkey

`--mode auto` runs unattended. `--mode human` puts you in the driver's seat
at the planning step. You can cross between them mid-run, in either
direction, without restarting:

- Press the hotkey (`h`) during any agent call and the mode flips at the
  next planning step.
- At the menu itself, type `a` (the line-oriented input is what the menu
  already owns, so the hotkey is paused while it is up).
- Ctrl-C stops the run from anywhere — the menu included — writing the
  checkpoint first; see [Cancelling a run](#cancelling-a-run-ctrl-c-and-resuming).
  At the menu, `q` does the same thing.

`proofs/ui/input.py` explains why the toggle is a keystroke in one place
and a line of input in the other.

### Proving a lemma yourself

At the menu, a number and Enter (or `w` for a lemma you write) means you
supply the proof:

1. **The proof.** Paste it, or type it, then press Enter to submit. A paste
   arrives whole, line breaks and all, because the prompt turns on the
   terminal's bracketed-paste mode. To type a line break, press Alt+Enter or
   Ctrl+J. Shift+Enter also works in terminals that send it differently from
   Enter (kitty, WezTerm, others with the keyboard protocol turned on); most
   send it as Enter, which submits. Backspace and Ctrl+U edit; Ctrl+D on an
   empty proof cancels. Without a terminal (a pipe, or Windows), the proof is
   read line by line and ended by a line holding a single `.`.
2. **The citations**, cited as the prover cites them: lemmas and references
   by their ids. Type them on one line separated by commas, e.g.
   `alice/3f9a2c1b4d5e6f70/5e02bd, eoinm/0c1d2e3f4a5b6c7d/c3917f, ref_3`, or paste the
   prover's JSON (`{"cited_lemmas": [...], "cited_references": [...]}`).
   An empty line means none. If you paste the prover's whole output object
   (with a `"proof"` field) as the proof, its citations are taken from it. A
   citation that does not parse, or that names a lemma that does not exist or
   is suspended, or a reference not in `references.md`, is reported as an
   error and asked for again; your proof is kept.
3. **Verification.** The three verifiers check your proof exactly as they
   check the prover's, seeing the same things: the statements of what you
   cite and nothing else. If all three accept, the lemma is committed to your
   run's stream under a new id and certified. If one rejects,
   its objection is shown and the menu comes back. Nothing is written.

## Sharing a conjecture

A conjecture is shared by putting its [workspace](#workspaces) in a git
repository that everyone pulls from and pushes to: `proofs init`, `git
init`, and commit `.proofs`, `.gitignore` and `conjectures/`. Anyone who
clones the repository has the workspace — the marker is committed — and
needs only their own `proofs config --global user.id`. `proofs` never runs
git itself: you pull before a session and commit and push afterwards. Every
run writes append-only streams of its own, so two users' work never lands
in the same file and a merge is only ever additions.

### Who writes what

| File | Written by | Committed |
| --- | --- | --- |
| `.proofs`, the proofs block of `.gitignore` | `proofs init` | yes |
| `conjecture.md`, `references.md` | the conjecture's maintainer | yes |
| `dags/<user_id>/<run_id>/*.jsonl` | that run: its lemmas, and a repair's reproofs | yes |
| `certificates/<user_id>/<run_id>/*.jsonl` | that run (`run`, `verify`, `repair`): its acceptances | yes |
| `computations/<user_id>/<run_id>/*.jsonl`, `code/<user_id>/<run_id>/` | that run: its code executions, and the scripts its lemmas rest on | yes |
| `refutations/<user_id>--<run_id>--<hex>--<verifier>--<time>.json` | the verifier's `verify` or `repair`; deleted by any user's `verify` or `repair` that accepts the lemma, and by `prune` | yes |
| `checkpoints/<user_id>/<run_id>/` | that user's runs: the checkpoint and the run's lock | no (gitignored) |
| `references/`, `comments.md` | each user, locally | no (gitignored) |

Each command reads every run's streams as one complete DAG, once, when it
starts — computing every hash, and reading every certificate and refutation
— and works from that in-memory DAG plus its own commits: a run does not see
what another process writes while it runs (pull, or run again, to pick that
up). A lemma's id, `<user_id>/<run_id>/<6 hex digits>`, is drawn by the code
when the candidate is proposed — by the planner, the lemma generator, the
reviser or you at the menu — and drawn again if the run has used it before.
It is unique everywhere without any coordination, since no other run shares
its run id, and the lemma keeps it from proposal to commit.

A stream is never rewritten except by `proofs prune`, so the only merge
conflict a stream can have is a prune on one machine against a run still
appending on another. It is rare, and resolved by hand: keep every record
line both sides have. Until it is, every command stops with the file and
line of the conflict marker rather than guess. The typical cycle:

```sh
git pull
proofs run NAME                                        # prove; writes this run's streams only
proofs verify NAME alice/3f9a2c1b4d5e6f70/7d1e90 --full     # check someone else's work
git add conjectures/NAME && git commit -m "..." && git push
```

### Streams

A stream file is JSON lines: a header first — the writer's `user.id`,
`user.name`, `user.email` and the run id — then one record per line, each
with a `"type"`. Files are numbered in hex, `000001.jsonl`,
`000002.jsonl`, …, at most 40 MB each; a record is never split between files.
A lemma record holds the lemma's id, statement, proof, the ids it cites
(`cited_lemmas`, `cited_references`), its computations, when it was
proved and the model that proved it. `proofs repair` never edits a lemma: it appends a **reproof** — the
whole new version under the same id, with `supersedes` naming the version it
replaces by that version's own content hash — to its own run's stream, until
`proofs prune` compacts it (see [Pruning](#pruning)). A
lemma's current version is the version no other version supersedes,
whatever order the lines are in.

More than one such version is a **fork**: two repairs of one version that
raced, or a repair of a version that a prune elsewhere has since deleted,
against the current one. Everyone decides a fork the same way:

- If every version states the same thing (up to whitespace), the lemma is
  valid if any of them is certified, and the current version is a certified
  one (the smaller lemma hash, if several are). Nothing citing the lemma is
  affected, since a citation pins only the statement. `proofs prune` keeps
  the certified version and deletes the rest.
- If the statements differ, the lemma is suspended, and every command
  reports it when it loads the DAG, until a person deletes all but one of
  the versions. `proofs prune` leaves it alone.

### Certificates and the Merkle hash

Every lemma has a **Merkle hash** (`proofs/core/merkle.py`). The statement
and the proof are each hashed on their own, whitespace-normalised and raw
(a stream stores them already normalised, so they are hashed as stored),
and the lemma hash is the SHA-256 of `{statement hash, proof hash, the sorted
statement hashes of everything it cites, its computations}`. A cited lemma
contributes the hash of its statement and a cited reference the hash of its
formal statement — not their ids, and not a cited lemma's proof. The hash therefore covers exactly
what the verifiers are shown. A new statement for a lemma changes the hash
of every lemma that cites it; a new proof under the same statement changes
only the lemma's own hash. Whether everything beneath a lemma is sound is
[suspension](#suspension)'s question, which walks the whole citation
closure. A citation cycle leaves the lemmas in it, and those citing their way
up to it, with no hash at all.
The computations are each script's hash and the output recorded for it
(see [Code tools](#code-tools-sage-python)). Hashes are never stored:
everyone who has the same files computes the same hashes.

A **certificate** (`proofs/core/certificates.py`) records an acceptance: one
record in the accepting run's certificate stream holding the lemma's id,
its hash at the time, the verifier's `user_id`, the model and the date. A
run writes one whenever all three verifiers accept a proof (all four, for a
proof with computations), the prover's or the operator's. `proofs verify`
writes one for each lemma it accepts. Every acceptance is a record of its
own; `proofs prune` merges duplicates (same lemma, hash, verifier and
model) into the earliest, with a `count`, and `proofs status` sums them. A
certificate is **valid** only while its hash
matches the lemma's current hash, so editing a lemma, or the statement of
anything it cites, quietly invalidates its certificates.

### Refutations

A **refutation** (`proofs/core/refutations.py`) records a rejection by `proofs
verify` or `proofs repair`: one file per rejection, holding the lemma's
id and hash, the verifier, the model, the date, the verifier's
justification, and the stream file holding the version it rejects. Rejections inside `proofs run` are not refutations; they
are feedback to that run's reviser. A refutation *counts* while its hash
matches the lemma's current hash. When the lemma changes, the refutation
goes stale but is kept, and its justification is shown to the verifiers
the next time the lemma is verified, so a known objection is not
forgotten. A verification that accepts the lemma deletes its refutations;
`proofs prune` deletes those of a lemma it removes, and a stale one once
the lemma's current version has a valid certificate.

### Suspension

A lemma is **suspended** (`proofs/core/suspension.py`) when any of these holds:

- it has no valid certificate;
- it has a refutation that counts;
- its versions state different things (a [fork](#streams) a person has to
  settle);
- it cites a suspended lemma.

Suspension is computed from the in-memory DAG and never stored. A suspended lemma is hidden from the planner and the prover, and the prover may not cite it; the run warns about each
one once. `verify`, `repair` and `status` still load suspended lemmas,
since verification is how they come back. A lemma whose run stopped
between committing it and certifying it is reported as such; run `proofs
verify` on it to give it its certificate.

### Verify, repair, prune and status

- **`proofs verify DIR LEMMA_ID [--full]`** runs the three verifier checks
  on a lemma's stored proof. With `--full`, everything the lemma cites is
  verified first, in dependency order, and the first rejection stops the
  run. An acceptance writes a certificate and deletes the lemma's
  refutations; a rejection writes a refutation. Exit code 0 means every
  lemma was accepted, 1 means one was rejected, and 3 means a verifier gave
  no verdict (the server failed). Nothing is written in that last case, so
  an outage never suspends anything.
- **`proofs repair path_to_refutation`** first re-verifies the lemma the
  refutation names, with the objection in front of the verifiers. If the
  proof is accepted, the refutation is dismissed and the repair stops.
  Otherwise the lemma is re-proved through the prover, verifiers and
  reviser — without decomposition, since it must keep its id — and the new
  proof is committed as a reproof under the same id, in the repair's own
  run stream. A new proof under the same statement leaves every dependent's
  certificate valid. If the reviser revised the statement, the lemmas
  citing it lose their certificates, so the repair then climbs the
  dependents in dependency order, re-verifying each one whose certificate
  no longer matches and re-proving any that are rejected. Repair refuses a
  [forked](#streams) lemma, the one named or one the climb reaches, since a
  reproof would supersede only one of its versions: if one of them is
  certified, `proofs prune` resolves the fork; otherwise delete all but one
  version by hand (the message names where they are), then repair.
- **`proofs status DIR LEMMA_ID`** lists, for the lemma and everything it
  cites, each lemma's current hash, the stream file of its current version,
  its valid certificates with how many times each was issued, and whether
  it is suspended; then whether the conjecture is settled. It writes
  nothing.

### Pruning

The streams only grow; **`proofs prune DIR [--with-dependents]`** is their
garbage collector. A lemma whose refutation still counts — the proof in the
files is the one a verification rejected, and no repair has replaced it — is
removed: every record of it (original and reproofs), every certificate of
it, and its refutation files. The lemmas citing it stay, suspended, so
`proofs repair` can still revive them; `--with-dependents` removes them too,
transitively (including the citers of a lemma an earlier prune removed).
Prune also shrinks the streams. A repaired lemma's current version is
rewritten in place, in the same file, as a plain lemma record — the file's
header still says who repaired it — and every other version of it (the
original, earlier reproofs, the losers of a fork) is deleted — a fork only
once the version it keeps is certified, and never one whose versions state
different things. Every certificate whose hash is not its lemma's current
hash is deleted, and duplicate certificates are merged into the earliest
with a `count`. A refutation that no longer counts is deleted once the
lemma's current version has a valid certificate. The computations and scripts are left alone. A stream a run on this machine is
still writing to is never rewritten: a lemma with a record in one is neither
removed nor compacted until the next prune. Prune also clears scratch — everything except a live run's, and
the executions an interrupted run's checkpoint still needs — and the run
directories with nothing to resume.

## Parallel mode

`--parallel N` runs N proof loops concurrently. Each loop is the same engine
the serial run uses — prover, the three verifiers, reviser, one
lemma at a time — and every loop reads and writes the same DAG. Two further
threads steer them: a **planner** that keeps a shared strategy current, and a
**lemma generator** that keeps the loops supplied with work.

```sh
proofs run conjectures/algebra_example --parallel 2
proofs run conjectures/algebra_example --parallel 3 --mode human
```

A bare `--parallel` is N=1: the parallel machinery with a single loop. The
loops never work on the same lemma — the buffer below makes that impossible —
a loop that has spent its budget stops on its own, and the run ends when the
last loop out stops, which is also what tells the planner and generator to
finish.

### Coding loops

With `--tools`, a parallel run takes one **code job slot** per loop that
may run code. `code.jobs` (see [Limits](#limits)) is the number of slots on
the machine, shared by every proofs command running on it — a slot is a
lock file in `$XDG_RUNTIME_DIR/proofs/slots/`, held until the command
exits — so `--parallel 3` with `code.jobs` 2 and no other run in progress
gets two **coding loops** (loop-0 and loop-1) and one loop without code,
whose prover, reviser and verifiers are never offered `run_code`. A
command that finds every slot taken says so: a parallel run goes on with
no coding loops; a serial run, `proofs verify` or `proofs repair` with
`--tools` stops, since its one loop needs one.

The lemma generator is told how many coding loops there are, and may mark
a candidate `"computation": true` when proving it needs a computer search
or an exact finite check. Only a coding loop claims a marked lemma — and a
coding loop takes marked lemmas before any other — while a loop without
code never does. At most one marked lemma per coding loop sits in the
buffer at a time; one marked beyond that is **held** outside the buffer and
enters it as a marked slot frees, so marked lemmas wait for the loops that
can prove them without crowding out the loops that cannot. (Held lemmas
are not in the checkpoint: after a restart the generator proposes afresh.)

### The shared possible-lemma list

The loops do not take lemmas from the planner directly. Between them sits a
fixed buffer of **N+5 slots** — the possible-lemma list. The generator is its
only writer: each round it proposes five candidate lemmas
(`proofs/agents/parallel/parallel_lemma_generator.md`), each given its own new
id by the code, and they go into the empty slots. The slots are also the flow control: a full buffer means
the loops are behind, and the generator waits for a free slot instead of
piling up work.

The buffer is a queue. An unattended loop claims the oldest unclaimed lemma —
first in, first out, by arrival order — and the claim is made under lock, so
two loops never work on one id. A claim released by a crashed loop keeps its
place in the queue. A
claimed lemma keeps its slot until it is settled — proved or given up after
`MAX_PROOF_ATTEMPTS` rounds — or the operator proves it from the menu.

The buffer is not the only source of lemmas: a reviser that decomposes a
hard lemma starts the prover on a smaller one, under a new id the code
draws for it. Every id is drawn once, so two loops never commit under one.

### The planner and the generator

- **Planner** (`proofs/agents/parallel/parallel_planner.md`) sets direction, not
  lemmas: a strategy summary and an ordered list of priorities — which gaps
  in the DAG to close next, and why. It runs once at start, and then whenever
  the DAG changes: a lemma added by *any* loop bumps the DAG version and
  marks the plan stale. If the DAG moves again *during* the planning call,
  the plan is installed already marked stale and the planner goes straight
  back around — no DAG change is lost, and the plan the loops are steering
  by was always written for the DAG as it stood.
- **Generator** (`proofs/agents/parallel/parallel_lemma_generator.md`) turns the
  plan into five concrete candidate lemmas per round. It sees the conjecture,
  the plan, the DAG, the buffer's current contents, and the reject list, so
  it never restates a lemma the loops already have, hold, or failed.

The split is deliberate: the planner re-reads the whole DAG and is the costly
call, so it runs only when the DAG moves; the generator is cheap and runs
continuously against the fixed buffer.

### Budgets

Each loop gets the full `--max-iterations` budget, spent one iteration per
lemma claimed. Resuming an in-flight lemma after a restart does not spend a
new one — the iteration was already spent when the lemma was claimed — and a
loop that dies holding a lemma still finishes it, budget or no. Failed
attempts go to one shared reject list that the steering planner and the
generator both read, so a route one loop found dead is dead for the whole
run.

### Human mode and the hotkey

`--mode human` takes the place of **loop-0**: the operator sits at loop-0's
menu while loops 1..N-1 run unattended. The menu shows the shared buffer —
each lemma marked with the loop that has claimed it — and everything the
serial menu can do works the same: prove a lemma yourself (the verifiers
check it before it goes into the shared DAG), send one through the proving
pipeline, write your own, or reject the buffer and send the planner back. The hotkey and the menu's `a` still hand loop-0 between human
and automatic mid-run; the other loops are always automatic. Quitting from
the menu stops the whole run, not just
loop-0 — the other loops' in-flight lemmas go into the checkpoint and resume
next time.

### Checkpointing and resuming

A parallel run writes a **version-2** checkpoint in its run's directory:
one slot per loop, each holding that loop's iteration budget and its lemma
in flight (or none) at the step it reached, plus the shared reject list,
the steering plan and the buffer — the waiting candidates and which loop
has claimed each. It is written atomically after every step. A serial (v1)
and a parallel (v2) checkpoint are not interchangeable: a parallel run
ignores a serial run's checkpoint and the other way round, so `--fresh`
starts the other kind of run.

Re-run the same command with the same `--parallel N` to resume: every
loop's budget continues rather than resets, the saved plan steers the run
from the start (the planner waits for the DAG to move before it plans
again), the buffer comes back with its claims, and a loop that was
mid-lemma resumes at the step it was on. The checkpoint records how many
loops wrote it, and a run with a different loop count does not load it (it
starts fresh from the DAG instead; the lemmas are kept either way). The
rest of [Cancelling a run (Ctrl-C) and resuming](#cancelling-a-run-ctrl-c-and-resuming)
holds unchanged.

## Project layout

This repository holds the program only; conjectures live in a
[workspace](#workspaces), a separate directory that `proofs init` lays
out wherever you choose. The repository:

```
proofs/
  agents/              # default agent prompts, installed with the package
    planner.md  prover.md  verifier_1/2/3.md  reviser.md   # one reviser for rejects and overflows
    parallel/                                    # parallel-mode prompts
      parallel_planner.md  parallel_lemma_generator.md
    parsing.md                                   # references parser prompt
    coding.md                                    # the run_code contract, shown with --tools
    coding_verifier.md                           # the fourth check, for proofs with computations
  cli/
    main.py            # the `proofs` command: subcommands and argument validation
    flags.py           # flags shared by run and repair (lemma window, tool mode)
  core/                # the engine: the DAG, its hashes and validity records, the run
    context.py         # the Context every part of a command is handed (no module globals)
    startup.py         # proofs run: flags, the backend, the run, then the serial or parallel loop
    streams.py         # the append-only JSONL streams: writing, reading, rewriting (prune)
    dag.py             # building the DAG from every run's streams, committing lemmas to yours
    checkpoint.py      # runs (ids, locks) and their checkpoints
    interrupts.py      # Ctrl-C: commit sections, the checkpoint on the way out
    merkle.py          # the Merkle hash of every lemma
    certificates.py    # certificate records: recording, validity, counting
    refutations.py     # refutation files: recording, loading, dismissal
    suspension.py      # which lemmas are not to be built on, computed from the files
    resolution.py      # whether the conjecture is settled, computed from the files
    prune.py, status.py  # proofs prune, proofs status
    migrate.py         # proofs migrate: format-1 workspaces to streams
    locking.py         # the run lock
  workspace/
    config.py          # the user's global config (~/.config/proofs/config): user.id, user.name, user.email
    settings.py        # the run's settings in .proofs-local.json (run.*, tools.*, llm.*)
    layout.py          # the workspace (init, root discovery, the .gitignore block), per-conjecture directories, prompt resolution
  loops/               # everything that calls a model
    reasoning.py       # reason() and extract(): one model call, the context-wall rescue
    json_repair.py     # reclaiming JSON from a reply that is not clean JSON
    prompts.py         # prompt assembly: the lemma window, the agents' views, fitting the window
    schemas.py         # response schemas
    proof.py           # the proof loop: prover -> verifiers -> reviser
    verifiers.py       # one verification step
    serial.py          # the serial run
    parallel.py        # the parallel run: buffer, steering planner and generator, N loops
    human.py           # human mode: the menu, the operator's own proof
    mode.py            # the auto/human toggle
    verify.py, repair.py  # proofs verify, proofs repair
  validity/            # the rules of verification: plain data in, decisions out
    steps.py           # what verification consists of
    verdicts.py        # reading a verdict, what it records, the re-execution policy
    citations.py       # checking a proof's citations
  llm/
    __init__.py        # connect(): make and probe the backend, settle the context window
    base.py            # the Backend interface the loops call, and the types that cross it
    openai_compat.py   # the OpenAI-compatible transport (llama.cpp or a hosted API)
  tools/
    __init__.py        # the registry: which tools an agent call is offered
    base.py            # Tool, ToolSet and the two tool-calling protocols
    lookup_lemmas/     # reading a proved lemma the lemma window lists by id only
    run_code/
      tool.py          # the run_code tool
      setup.py         # --tools: checking tools and limits, the sandbox relaunch
      codetools.py     # the code tools (sage, python), tools.lock, finding them on PATH
      limits.py        # code limits: defaults, the per-machine files, the memory check
      sandbox.py       # bubblewrap: the outer sandbox around proofs, the inner one per script
      execution.py     # running one script: scratch directory, watchdog, RESULT
      computations.py  # computations in the DAG: code_runs, code/, computations/, what verifiers see
  references/
    parsing.py         # standalone references/ → references.md parser
  export/
    latex.py           # the LaTeX export
    command.py         # proofs export
  ui/
    plain.py           # line-by-line output
    live.py            # the live terminal block (rich): the status board under the log
    status.py          # the status board: what each agent is doing
    input.py           # terminal input: the hotkey, the human menu's readers
pyproject.toml             # the package and the `proofs` console script
flake.nix                  # Nix dev shell, package and app (`nix run .`)
```

A workspace, as `proofs init` and `proofs new` lay it out:

```
.proofs                # the workspace marker (format = 2)
.gitignore             # your own rules, plus the proofs block proofs init writes
.proofs-local.json     # gitignored: this machine's code limits and run settings, from proofs init
conjectures/
  algebra_example/
    conjecture.md          # required: the statement
    dags/<user_id>/<run_id>/000001.jsonl          # committed: each run's lemmas and reproofs
    certificates/<user_id>/<run_id>/000001.jsonl  # committed: each run's acceptances
    checkpoints/<user_id>/<run_id>/  # gitignored: each run's checkpoint and lock
    refutations/           # committed: one file per rejection by verify or repair (created on first write)
    references/            # maintainer-local source files (.tex/.md/plain text), gitignored
    references.md          # committed: parsed theorem-level results (strict JSON array)
    comments.md            # optional, gitignored: this user's private notes for the planner
    prover.md              # optional per-conjecture prompt override
    flake.nix, flake.lock  # optional, committed: the maintainer's code tools
    tools.lock             # committed with them: the tools and versions runs may use
    limits.json            # optional, gitignored: this machine's code limits for this conjecture
    scratch/<user_id>/<run_id>/  # gitignored: every code execution of a run (script, output, run.json)
    code/<user_id>/<run_id>/       # committed: the scripts that proofs rest on, named by hash
    computations/<user_id>/<run_id>/000001.jsonl  # committed: each run's executions of them
```

### Workspaces

A workspace is a directory holding a `.proofs` marker and a `conjectures/`
directory. `proofs init [DIR]` makes one (`DIR` defaults to the current
directory, and is created if it is not there):

- it creates `DIR/conjectures/`;
- it writes `DIR/.proofs`, which records the layout version, `format = 2`;
- it writes `DIR/.proofs-local.json`, this machine's [code limits](#limits)
  and [run settings](#settings), set to the built-in defaults for you to
  edit;
- it writes the proofs block of `DIR/.gitignore`: the rules that keep each
  user's local files — `references/`, `comments.md`, checkpoints, lock and
  temp files, `.env` — out of git. The block sits between
  `# >>> proofs >>>` and `# <<< proofs <<<`; an existing block is replaced
  in place, a `.gitignore` without one has it appended, and lines outside
  it are never touched, so keep your own rules outside the markers. The
  rules live in one place, `GITIGNORE_RULES` in `workspace/layout.py`, so
  re-running `proofs init` brings a workspace's block up to date.

Re-running `proofs init` is safe: an existing marker, limits file and
conjectures are left alone, and only the block is rewritten. Run inside an
existing workspace — in a subfolder, or in its `conjectures/` — it refuses,
since a nested workspace is almost certainly a mistake. Outside a git
repository it says so, with a hint to run `git init` to share the
workspace.

Every other command finds the workspace by walking up from the current
directory to the first folder with a `.proofs`, as git finds `.git`, so a
bare conjecture name works from anywhere inside it. Outside a workspace:

- `proofs new` stops with `not in a proofs workspace; run proofs init`;
- `run`, `parse`, `prune`, `verify`, `repair`, `status` and `export` keep
  working given an explicit path to a conjecture or refutation,
  and give the same error for a bare name;
- `proofs config` is unaffected: the config is per user, not per workspace.

A directory laid out before workspaces — a `./conjectures/` with no
`.proofs` above it — is still used as the root when commands are run from
it, with a warning to run `proofs init` there.

### Migrating a format-1 workspace

Format 1 kept one file per user and kind: `dags/<user_id>_dag.json`,
`certificates/<user_id>.jsonl`, `runs/<user_id>.jsonl` and
`code/<user_id>/`, with lemmas identified by the pair (user, lemma id).
Every command refuses a conjecture still laid out that way, and says to run
**`proofs migrate [DIR]`**, where `DIR` is the workspace root. Commit your
work first; then review the result (`git status`) and commit it.

- Each user becomes one run per conjecture, with a run id derived from the
  user id, so everyone who migrates the same files gets the same files, and
  an interrupted migration picks up where it stopped when run again.
- A lemma's new id ends in its old one: `lemma_56` in Spark's file becomes
  `Spark/<run_id>/lemma_56`. Statements and proofs are copied untouched, so
  a proof's "by lemma_56 (Spark)" still reads onto the cited lemma's id;
  citations are rewritten to the new ids.
- A certificate that was valid is issued again under the new hash, which
  covers the same statement, proof and cited statements the verifier
  accepted. It keeps its verifier, model, date and count, in the verifier's
  run. A certificate that was not valid is dropped.
- A refutation that still counted is written again with the new id and
  hash and the stream file of the version it rejects; one that no longer
  counted is dropped.
- `runs/` becomes each user's computation stream, counts kept, and
  `code/<user_id>/` moves to `code/<user_id>/<run_id>/`.
- The old files, old checkpoints, locks and temp files are deleted, the
  marker goes to `format = 2`, and the `.gitignore` block is rewritten.

### Conjectures and the shared DAG

`DIR` names a conjecture directory: a name under the workspace root's `conjectures/`, or a path
to one, holding a `conjecture.md`. Every run's lemmas live in its own
stream, `dags/<user_id>/<run_id>/`, and every command builds the complete
DAG from all of them (see [Streams](#streams)). Point a second user, or a
second model, at a conjecture already under way and it continues from the
lemmas already proved: a lemma proved by a 27B model and certified is no
less proved when a 122B model arrives. The cost is that this is
collaboration, not a controlled comparison — results are entangled from the
first shared lemma.

A lemma record stores exactly:

| Field | Meaning |
| --- | --- |
| `type` | `"lemma"`, or `"reproof"` for a version written by `proofs repair`. |
| `id` | The lemma's id, `<user_id>/<run_id>/<6 hex digits>`: the run that proposed it, and a random part. |
| `statement` | The lemma's statement. |
| `proof` | The proof of this version. |
| `cited_lemmas` | The ids of the lemmas the proof cites. |
| `cited_references` | The ids of the references the proof cites. |
| `code` | The computations the proof rests on, if any (see [Code tools](#code-tools-sage-python)). |
| `proved_at` | When this version was written (the lemma window's recency). |
| `model` | The model that wrote this version's proof; absent when not known (a lemma `proofs migrate` converted, or one resumed from a checkpoint older than the field). |
| `supersedes` | A reproof only: the content hash of the version it replaces. |

Who wrote a version is the header of the stream it is in; which model, its `model`. The DAG built at
startup keeps every lemma's statement, its hashes, citations and where its
current version is — not the proofs, which are read from the stream when a
prompt needs one. Only the statement, the proof, the cited statements and
the computations go into the [Merkle hash](#certificates-and-the-merkle-hash).

Prompt files are looked up in the conjecture directory first, then in the
defaults shipped with the package (`proofs/agents/`), so a conjecture that
needs a special prompt (e.g. a prover primed for Brauer groups) can carry
one without forking the defaults.

### The status board

On a terminal, `proofs run` keeps a small block at the bottom of the screen
while the log scrolls above it: one line about the run (the conjecture, the
model, the run id, the iteration, how long it has been going) and one row
per agent — the serial run's loop, or a parallel run's planner, generator
and each loop — with what it is doing (`planner`, `prover`, `verifier 2`,
`reviser`, `waiting for a lemma`, ...), the lemma it is on, the prover
round, how long it has been at it, and the last thing that happened to it
(`verifier 2/3 rejected`, `committed alice/3f9a2c1b4d5e6f70/5e02bd`). Nothing is streamed from
the model, so the time column is what moves during a long call. The block
steps aside while the human menu, a pasted proof or a y/N is on screen,
and is cleared when the run ends. `--ui plain` turns it off; it is off
anyway when the output is not a terminal, and when `rich` is not installed
(`nix develop` provides it).

### Cancelling a run (Ctrl-C) and resuming

Every `proofs run` is a **run** with an id of its own (16 hex digits, shown
in the banner) and a directory, `checkpoints/<user_id>/<run_id>/`, local and
gitignored, holding its checkpoint and a lock the running process holds.
`proofs verify` and `proofs repair` are runs too, with a lock and no
checkpoint; a run's code executions go to `scratch/<user_id>/<run_id>/`.

Ctrl-C stops the run and leaves the checkpoint. If Ctrl-C lands while an
accepted lemma is being committed and certified, the run finishes that step
first (press Ctrl-C again to force), so a lemma is never left committed
without its certificate. The checkpoint records what a cancelled run would
otherwise lose: where the iteration budget stood, the latest plan, the
planner's reject list (`rejected_lemmas`: the statements of the last
`run.max_rejected_lemmas` lemmas that spent every prover round unproved),
and the lemma in flight — its prover round, the step it reached (`prove`,
`verify` or `revise`), the prover's output for that round, and the verdict
of every verifier that finished. It is written after every step, and
deleted when the conjecture resolves; any other exit keeps it, with nothing
in flight after a clean one.

Re-run the same command to pick up where the run stopped. With no flag,
`proofs run` continues your most recent run on the conjecture — unless
another process is running it, in which case a new run starts beside it;
`--fresh` starts a new run, and `--resume RUN_ID` continues a given one.
The budget continues rather than resets, the saved plan stands in for the
planner, and a lemma in flight resumes at its step: the verifiers that had
not finished, or the reviser, without re-running the prover. A run may
resume on another model. A saved proof keeps the model that wrote it, which
the committed lemma names. A saved verdict counts only while the proof's
hash is the one it checked and it came from the model the run uses — the
model the server reports (a llama-server's weights file, whatever
`--model` says), or the `--model` request before a hosted API has replied;
otherwise that verifier runs again, since a certificate names one model.
A saved proof that no longer stands (a citation that stopped resolving, a
script gone from the scratch directory) sends the round back to the prover
at no cost to its budget. Only the interrupted call itself is lost.

A bad checkpoint degrades instead of crashing: an unreadable or incoherent
one is discarded and the run starts fresh; a checkpoint whose in-flight
lemma is already in the DAG drops only the in-flight part and resumes as a
normal iteration; one whose iteration stands past `--max-iterations` is
reported and kept — raise the budget and re-run to continue from it.

### Settings

The run's tunables live beside the code limits in the workspace's
gitignored `.proofs-local.json` (`proofs init` writes every key at its
default). A flag, where there is one, beats the file for one run.

| Key | Default | Flag | Meaning |
| --- | --- | --- | --- |
| `run.max_iterations` | -1 | `--max-iterations` | Loop iterations before the run stops; per loop in a parallel run. `-1` runs until the conjecture is settled or the run is interrupted. |
| `run.max_proof_attempts` | 3 | `--max-proof-attempts` | Prover rounds one iteration may spend on a lemma. |
| `run.max_rejected_lemmas` | 4 | `--max-rejected-lemmas` | Rejected lemmas the planning agents are shown. |
| `run.max_loop_restarts` | 3 | | Parallel mode: restarts in a row of a crashed thread before it is given up. |
| `tools.max_rounds` | 3 | | Tool rounds one agent call may spend before it must answer. |
| `tools.code_rounds` | 8 | | The same, for a call offered `run_code`. |
| `tools.max_lookup` | 10 | | Lemmas one `lookup_lemmas` call may return. |
| `llm.max_retries` | 3 | | Attempts at one model call. |
| `llm.max_compaction_passes` | 2 | | Compaction-and-resume passes one truncated call may get. |

### References

The conjecture's maintainer keeps a `references/` directory beside
`conjecture.md`: chapters, surveys or papers, as `.tex`, `.md` or plain
text, that the agents may lean on. In the distributed setup the two sides
of that directory have different owners:

- **`references/` (the input) is the conjecture maintainer's local
  material.** It is gitignored and never committed: it may be large, and it
  is the maintainer's working copy of the sources.
- **`references.md` (the output) is a committed artefact, maintained by the
  maintainer and committed with the conjecture.** Only the maintainer runs
  the parser — a standalone tool, not part of the proving loop:

  ```sh
  proofs parse conjectures/algebra_example
  ```

  It reads every parsable file in `references/`, asks the model (via
  `proofs/agents/parsing.md`) for the theorem-level results, and writes them to
  `references.md` beside the DAG as a **strict JSON array** — one object per
  result with `id`, `slogan`, `formal statement` and `reference` (where it
  appears in the file). The maintainer then commits and pushes the file
  (the tool reminds them to).
- **Every other user pulls `references.md` and never regenerates it.**
  Nothing in `run`, `verify` or `repair` writes the file; the loop only
  reads it. A result they need that is missing is *requested from the
  maintainer*, who parses it in and commits the new entry — and the loop
  says so in the log when a proof cites a reference id that is not in the
  collection, instead of burning rounds retrying the same citation.
  (Running `proofs parse` in a clone without the maintainer's
  `references/` is a clean no-op that leaves the pulled file untouched.)

`main.py` feeds that file to the loop at three granularities:

- **Planner and reviser** — an `id` + `slogan` shortlist, and only while the
  collection is small: below `PLANNER_REFERENCE_LIMIT` (100) entries each
  gets every result, at or above it none. A shortlist of hundreds of
  slogans costs more than it returns, and the prover sees the full
  collection anyway, so a named result the planner misses is one round
  away, not lost. A strategy for large collections is deliberately not
  built yet.
- **Prover** — the whole collection unconditionally (id, slogan, formal
  statement). It may cite any entry by `id` in `cited_references` and
  rely on it without proving it, exactly like a proved lemma (a cited lemma
  goes in `cited_lemmas` instead, by its id).
- **Verifier** — only the formal statements of the results the proof
  actually cites, merged with the cited DAG lemmas into one set. The rest
  of the collection is withheld on purpose: a proof that leans on a result
  it never cited must read as an unjustified leap, and the verifier is
  told that citing a reference is legitimate while proving it inline is
  not.

Two properties of the parser keep citations valid as the collection grows —
which is what makes a shared, committed file safe to keep re-parsing:

- **Reference ids are stable across re-parses.** `ref_N` numbers are
  assigned by the tool, never by the model; a result whose formal statement
  matches an existing entry keeps its id, so a lemma that cites `ref_3` in
  its `cited_references` still means the same thing after the maintainer
  adds a chapter to `references/` and re-parses.
- **Entries are never deleted.** A re-parse is a union: entries from a
  previous `references.md` whose statements were not re-derived this run are
  kept, so a flaky parse of one file cannot silently delete a result another
  user's proof cites. An empty `references/` directory never touches an
  existing `references.md`. There is no clear mode: to reset the collection,
  delete or rewrite `references.md` (lemmas that cited its entries will warn
  about the dangling ids) — deleting it and re-running `proofs parse` is the
  cleanest way to start over. Note that a cited reference's formal
  statement is part of the citing lemma's Merkle hash, so changing an
  entry's statement invalidates the certificates of every lemma that cites
  it.

A run without `references/` or `references.md` runs with no references
section in any prompt, which is also the failure mode for a corrupt
`references.md` (it warns once and is ignored).

### Comments (private hints to the planner)

A conjecture directory may hold a `comments.md` beside its
`conjecture.md`: free-form notes on possible approaches to a proof or a
counterexample — routes to try, objects or theorems worth using, or dead
ends to avoid. No tooling touches it; you just edit it. On every planning
step the loop hands the file's contents to the planner verbatim as a
**Human comments** section, and `proofs/agents/planner.md` tells the planner how
to read it: weigh the suggestions — when a comment points at a viable route
the state of the proof allows, one of the five candidates should be a
concrete version of it — but treat them as suggestions, not instructions,
since a comment may be mistaken or stale.

The file is gitignored: each user keeps their own `comments.md`, and
comments are never committed or shared. A clone you pull therefore carries
no one else's notes, and yours never reach anyone else's run — the file is
per user the way the streams are, but by absence of sharing rather than
by a `<user_id>` in the name. The planner is the only agent that sees it
(the prover, verifiers and reviser are deliberately left to judge the
mathematics on its own), and a conjecture directory without a `comments.md`
runs with no comments section in the planner's prompt, the same way the
reference blocks degrade. The run banner announces the file when it
is present.
