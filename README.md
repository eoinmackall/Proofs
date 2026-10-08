# Multi-Agent Theorem Prover

A Python program that drives a small team of LLM agents to iteratively prove
(or refute) a mathematical conjecture. The agents work in a loop —
**planner → selector → prover → verifier → reviser** — and accumulate a proof
as a DAG of lemmas. (In `--mode human`, the operator takes the selector's
place at the menu.)

The system is distributed and coordinated through git. Several people can
work on one conjecture: each user's lemmas go in their own DAG file,
`conjectures/NAME/dags/<user_id>_dag.json`, and every run reads all the
users' files as one complete DAG. Every acceptance is recorded as a
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
   and any previously rejected attempts, and proposes
   `PLANNER_CANDIDATES` (5) candidate lemmas, best-first, working either
   toward a proof of the conjecture or — where it looks false — toward a
   concrete counterexample to it.
2. **Selection** picks exactly one candidate:
   - `--mode auto`: `screen_candidates()` filters out candidates whose ids are
     already in the DAG, and when more than one survives, the **selector**
     (`selector.md`) weighs them all and picks the one it judges most likely
     to come through the prover and the verifiers. The planner's best-first
     order is the fallback: it is taken when the selector fails or when only
     one candidate survives.
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
   steps, one per agent; a lemma enters your DAG file only after all three
   have accepted it, and the acceptance is recorded as a certificate in your
   certificate file. The three prompts are specialized: `verifier_1.md` checks
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
   to prove as stated (decompose it into a smaller lemma — a fresh id and a
   sub-statement the last proof left unjustified — and start the prover on
   that with a fresh budget). After `MAX_PROOF_ATTEMPTS` (default 3) prover
   rounds the lemma is given up and the planner is asked again. When the
   prover instead overflows the context window, the same reviser does the
   same job from the partial proof — see
   [When a call hits the context wall](#when-a-call-hits-the-context-wall).

### When the conjecture is settled

No agent can declare the conjecture proved. It is settled by a lemma, and
two lemma ids are reserved for that lemma:

- `conjecture`: the conjecture itself.
- `conjecture_negation`: that the conjecture is false, by an explicit
  counterexample.

The planner (or, in parallel mode, the lemma generator) attempts the
conjecture by proposing a candidate under one of these ids. Its statement
is always written by the code, not by the model: the verbatim text of
`conjecture.md`, or that text after the fixed prefix "The following
conjecture is false. Exhibit an explicit counterexample…". From there it is
an ordinary candidate. The selector weighs it against the others with no
priority, the prover assembles its proof from the DAG, and the three
verifiers check it. The reviser may not revise a pinned statement, and may
not decompose a lemma onto a reserved id.

Whether the conjecture is settled is computed, never stored
(`proofs/resolution.py`). It is settled while some lemma in any user's DAG
file meets both conditions:

- Its statement is exactly the pinned text, up to whitespace.
- It is not suspended, so it holds a valid certificate and everything it
  cites, all the way down, is certified and unrefuted.

The test is on the statement, so a proof committed under a renamed id
(`conjecture_2`) counts too. The serial run checks at the top of every
iteration and once after the last; the parallel planner checks on every DAG
change. Either way, a run stops when another user's pulled DAG settles the
conjecture. Editing `conjecture.md`, or a refutation or repair anywhere
below the settling lemma, reopens the conjecture on its own. A DAG that
proves both the conjecture and its negation
settles nothing: the run warns and carries on.

### Thinking vs. structured output

The prompts instruct each model to emit strict JSON directly, so every
response is first parsed as-is; a schema-constrained *extraction* call only
runs as a fallback. The underlying tension there — on llama.cpp, reasoning
silently voids the grammar — is handled inside `proofs/llm_backend.py`, which
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

One transport, probed at startup (`proofs/llm_backend.py`): the
OpenAI-compatible `/v1/chat/completions` endpoint. The same code serves two
backends:

- **llama.cpp** — a local `llama-server`. There the grammar suppresses
  thinking, so reasoning is disabled for schema calls.
- **a hosted API** — any OpenAI-compatible provider (OpenAI, OpenRouter,
  DeepSeek, ...), pointed at by `--host` with its base URL and authenticated
  with `--api-key` (or `$LLM_API_KEY`). A supplied key is checked against the
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
backend is taken on faith and the banner warns when it contradicts the probe
(a forced llama.cpp against a server with no `/props` route, or a forced
standard OpenAI against one that has it). The probe's *measurements* — the
context ceiling and thinking support — are kept either way, because they
describe the server that is actually there.

**What `--model` does, and which model gets recorded.** `--model` is sent
as the `model` field of every request, and that is all it does. A hosted
API serves the model it names. A llama-server ignores it: it answers with
whatever model it loaded at launch, and names its `--alias` in the reply.

Certificates and refutations therefore record the model that actually did
the work, not the `--model` request:

- **On a llama-server**, the file name of the loaded model, read off
  `/props` at startup (e.g. `Qwen3.8-27B-UD-Q3_K_XL.gguf`). It is printed
  in the banner as `loaded=` beside `model=`, the name `--model` sent.
- **Anywhere else**, the model named in the reply's own `model` field — for
  a hosted API, the model `--model` selected, usually with its version
  suffix (`gpt-4o` → `gpt-4o-2024-08-06`).
- **When the server names nothing**, the `--model` name.

The `/v1/models` list is not used for naming: on a hosted API it is the
provider's whole catalogue.

Model capabilities (context size, thinking support) are *profiles*, probed
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
`requests` works, run from the repository root as `python -m proofs ...`.

## Usage

### The `proofs` command

One command, nine subcommands:

| Subcommand | Takes | Does |
| --- | --- | --- |
| `proofs config KEY [VALUE]` | a config key, an optional value | Gets or sets the user's config, git-style: `user.id`, `user.name`, `user.email`. |
| `proofs new NAME` | a directory name | Lays out a blank conjecture, `conjectures/NAME/`: empty `references/`, `dags/` and `certificates/` directories, an empty `conjecture.md` and `comments.md`, and a `references.md` holding the empty collection `[]`. Refuses a `NAME` that already exists. |
| `proofs run DIR` | `DIR`, then the loop's options | Runs the proof loop on the conjecture. |
| `proofs parse DIR` | `DIR`, then the parser's options | The maintainer's tool: parses `DIR`'s `references/` into the committed `references.md`. |
| `proofs verify PATH lemma_id [--full]` | a user's DAG file, a lemma | Verifies the lemma; with `--full`, what it cites too, cited lemmas first. |
| `proofs repair path_to_refutation` | a refutation file | Repairs the lemma the refutation names. |
| `proofs prune DIR` | a conjecture directory | Drops the current user's stale certificates. |
| `proofs status PATH lemma_id` | a user's DAG file, a lemma | Lists the valid certificates in the lemma's citation closure, then whether the conjecture is settled. |
| `proofs export PATH [lemma_id] -o OUT_DIR` | a DAG file or a conjecture directory, an optional lemma, the output directory | Writes a LaTeX document of the DAG, in dependency order, into OUT_DIR. |

`DIR` is a conjecture directory. `PATH` is a user's DAG file,
`conjectures/NAME/dags/<user_id>_dag.json`, whose name supplies the lemma's
`user_id` — a lemma is addressed as `(user_id, lemma_id)`; for `export`,
`PATH` may be the conjecture directory itself, in which case the complete DAG
is written. `status` lists the valid certificates in the named lemma's
citation closure; `export` writes the DAG's LaTeX document — the complete
DAG, a user's lemmas, or a lemma and everything it cites — in dependency
order (dependencies first), into the directory `-o`/`--out` names (created
if it is not there yet — the document is the user's artifact, and never
lands in the project on its own): `proof.tex` for the complete DAG,
`<user_id>_proof.tex` for a user's lemmas, and
`<user_id>--<lemma_id>_proof.tex` for a lemma and its dependencies.

Every run acts as a **user**, named by a `user.id` that identifies its DAG
file, its certificates and its refutations. The user's identity —
`user.id`, `user.name` (a full name) and `user.email` — is stored in
`~/.config/proofs/config` (or `$XDG_CONFIG_HOME/proofs/config`), outside the
repository, so it is never committed. It is set and read git-style, through
the `config` subcommand: `proofs config --global user.id ID` sets (likewise
`user.name` and `user.email`), `proofs config user.id` prints, an empty
value unsets, and `proofs config --list` lists the set keys. A run without
a `user.id` fails and names the command to set it; proofs does not ask for
one, as git does not.

The identity is also stamped at the top of the user's generated files, so
a file read out of context still says whose it is: the user's DAG file
holds `user.id`, `user.name` and `user.email` as the first entries of its
object, and the user's certificate file holds them as its first line,
before the certificates. The stamp is the owner's to write — a run updates
it from the owner's config when it writes the owner's own file, and keeps
it as it stands when it writes another user's file (the repair's one
cross-user write), since it does not hold that user's name and email. The
certificate's readers skip the line; it names no lemma.

```sh
proofs run conjectures/algebra_example
proofs run conjectures/algebra_example --model Qwen3.5-122B-Q4_K_M
proofs run conjectures/algebra_example --no-verbose --max-iterations 25
proofs run conjectures/algebra_example --mode human
proofs run conjectures/algebra_example --parallel 2
proofs run conjectures/algebra_example --parallel 3 --mode human

# or against a hosted API instead of a local llama-server
LLM_API_KEY=sk-... proofs run conjectures/algebra_example \
    --host https://api.openai.com --model gpt-4o

# once, before proving: parse the conjecture's reference collection
proofs parse conjectures/algebra_example

# the DAG as a LaTeX document, in dependency order: the complete DAG, or
# one user's lemmas, or a single lemma with everything it cites, written
# into the directory -o names (created if it is not there yet)
proofs export conjectures/algebra_example -o ~/tex
proofs export conjectures/algebra_example/dags/alice_dag.json -o ~/tex
proofs export conjectures/algebra_example/dags/alice_dag.json lemma_42 -o ~/tex

# start a new conjecture, then write its statement into conjecture.md
proofs new standard_d

# on a fresh machine, set the identity once, git-style
proofs config --global user.id eoin
proofs config --global user.name "Eoin Mackall"
proofs config --global user.email eoin@example.com
```

`run` and `parse` both take one argument, the conjecture directory: a name
under `conjectures/` or a path to it — anything after it is passed through
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
| `DIR` | — | The conjecture directory: a name under `conjectures/` holding a `conjecture.md`, or a path to it. Taken by `run` and `parse`. |
| `--model NAME` | `qwen3.8-27b` | Sent as the `model` field of every request. On a hosted API, the model to use (e.g. `gpt-4o`); a llama-server ignores it and serves the model it loaded. Not what certificates record — see [Backend](#backend). |
| `--host URL` | `http://localhost:8081` | OpenAI-compatible base URL — a local llama-server, or a hosted API's base (e.g. `https://api.openai.com`, without the `/v1` suffix — it is added automatically). `$LLAMA_HOST` overrides the default. The default is 8081 rather than llama-server's own 8080, which other local services (open-webui, for one) often hold. |
| `--api-key KEY` | `$LLM_API_KEY` | Sent as `Authorization: Bearer <key>`. Needed for a hosted API; leave unset for a plain local llama-server. |
| `--backend {auto,llamacpp,openai}` | `auto` | Which dialect of the OpenAI-compatible endpoint to speak: the full llama.cpp option set (with thinking suppressed for schema calls), or the standard OpenAI set. `auto` takes the probe's answer; the other two force it, for when the probe can't see the real server. |
| `--max-iterations N` | 10 | Loop iterations before giving up. |
| `--max-proof-attempts N` | 3 | Prover rounds per lemma before it is given up. |
| `--lemmas-all N` | 25 | How many of the most recently proved lemmas (anyone's) the agents see in full; every other proved lemma is listed by `user_id` and `lemma_id` only. `-1` shows all. Also taken by `proofs repair`. |
| `--my-lemmas N` | 25 | How many of your own most recently proved lemmas the agents also see in full, overlap with `--lemmas-all` shown once. `-1` shows all. Also taken by `proofs repair`. |
| `--lemmas-common K` | 10 | How many *common lemmas* the agents also see, statement only: the lemmas outside the two recency windows that the shown lemmas cite most often (direct citations only). Ties are broken in a random order fixed for the run. A lemma nothing shown cites never ranks. `-1` shows every cited lemma. Also taken by `proofs repair`. |
| `--tool-mode {auto,json}` | `auto` | How agents call tools such as `lookup_lemmas`. `auto` uses the server's native tool calling and falls back to a JSON protocol the first time the server refuses it (llama-server needs `--jinja` for native calls); `json` uses the protocol from the start. Also taken by `proofs repair`. |
| `--fresh` | off | Ignore and delete your checkpoint (`dags/<user_id>_dag.checkpoint.json`), restarting the iteration budget from 1. The DAG files are kept. |
| `--parallel [N]` | off | Run N proof loops in parallel around one shared DAG, steered by one planner and one lemma generator. Bare `--parallel` is N=1. `--mode human`, when set, takes the place of one of the loops (loop-0). See [Parallel mode](#parallel-mode). |
| `--num-ctx N` | the server's own (probed) | Context window in tokens. On a local llama-server: defaults to the `-c` it was launched with, as probed — lower it if VRAM is tight — and an explicit value is clamped to that ceiling. On a hosted `--host` API nothing is probed, so the default is 65536 and an explicit value replaces it. Budgeting above the server's own window is caught at once from the server's 400, which names its real limit and the `--num-ctx` to re-run with. |
| `--mode {auto,human}` | `auto` | Automated selection vs. human menu at the planning step. |
| `--hotkey KEY` | `h` | Keystroke that toggles auto/human mid-run (pass `''` to disable). |
| `--verbose / --no-verbose` | on | Console logging. |

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
by `user_id` and `lemma_id` only, and the agent can read any of them with
the `lookup_lemmas` tool (statement and citations, never a suspended
lemma). When the window holds every lemma, no index and no tool are sent
and the prompts carry neither.

Tools live in `proofs/tools.py`: each is a name, a JSON Schema for its
arguments and a handler. `reason()` runs the loop — at most
`MAX_TOOL_ROUNDS` (3) rounds per call, `MAX_LOOKUP` (10) lemmas per
lookup — using native tool calls where the server supports them and a JSON
protocol otherwise (`--tool-mode`). If a call hits the context wall after
using tools, the continuation carries the tool results in its task.

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

`proofs/interaction.py` explains why the toggle is a keystroke in one place
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
2. **The citations**, cited as the prover cites them: lemmas by owner and id,
   references by id. Type them on one line separated by commas, e.g.
   `alice:lemma_5, eoinm:lemma_12, ref_3`, or paste the prover's JSON
   (`{"cited_lemmas": [{"user_id": ..., "lemma_id": ...}], "cited_references": [...]}`).
   An empty line means none. If you paste the prover's whole output object
   (with a `"proof"` field) as the proof, its citations are taken from it. A
   citation that does not parse, or that names a lemma that does not exist or
   is suspended, or a reference not in `references.md`, is reported as an
   error and asked for again; your proof is kept.
3. **Verification.** The three verifiers check your proof exactly as they
   check the prover's, seeing the same things: the statements of what you
   cite and nothing else. If all three accept, the lemma is committed to your
   DAG file with you as its `last_prover_id` and certified. If one rejects,
   its objection is shown and the menu comes back. Nothing is written.

## Sharing a conjecture

A conjecture is shared by putting its directory in a git repository that
everyone pulls from and pushes to. `proofs` never runs git itself: you pull
before a session and commit and push afterwards. The files are laid out so
that two users' work almost never lands in the same file.

### Who writes what

| File | Written by | Committed |
| --- | --- | --- |
| `conjecture.md`, `references.md` | the conjecture's maintainer | yes |
| `dags/<user_id>_dag.json` | that user's runs (and `proofs repair`, see below) | yes |
| `certificates/<user_id>.jsonl` | that user's runs, `verify` and `repair` | yes |
| `refutations/<user_id>--<lemma_id>--<verifier>--<time>.json` | the verifier's `verify` or `repair`; deleted by any user's `verify` or `repair` that accepts the lemma | yes |
| `dags/<user_id>_dag.checkpoint.json` | that user's runs | no (gitignored) |
| `references/`, `comments.md` | each user, locally | no (gitignored) |

Each command reads every user's DAG file as one complete DAG, but a run
writes new lemmas only to your own file. A lemma is identified by the pair
`(user_id, lemma_id)`: ids are unique only within one user's file, so
proofs cite lemmas as `{"user_id", "lemma_id"}` pairs. The typical cycle:

```sh
git pull
proofs run conjectures/NAME            # prove; commits to your files only
proofs verify conjectures/NAME/dags/alice_dag.json lemma_7 --full   # check someone else's work
git add conjectures/NAME && git commit -m "..." && git push
```

### Certificates and the Merkle hash

Every lemma has a **Merkle hash** (`proofs/merkle.py`): the SHA-256 of its
whitespace-normalised statement and proof together with the hashes of the
statements of everything it cites. A cited lemma contributes the hash of
its statement and a cited reference the hash of its formal statement — not
their ids, and not a cited lemma's proof. The hash therefore covers exactly
what the verifiers are shown. A new statement for a lemma changes the hash
of every lemma that cites it; a new proof under the same statement changes
only the lemma's own hash. Whether everything beneath a lemma is sound is
[suspension](#suspension)'s question, which walks the whole citation
closure. A citation cycle leaves the lemmas in it with no hash at all.
Hashes are never stored: everyone who has the same files computes the same
hashes.

A **certificate** (`proofs/certificates.py`) records an acceptance: one
line in the verifier's own `certificates/<user_id>.jsonl` holding the
lemma's pair, its hash at the time, the verifier's `user_id`, the model,
the date and a count. A run writes one whenever all three verifiers accept
a proof, the prover's or the operator's. `proofs verify` writes one for
each lemma it accepts.
Accepting the same lemma again with the same model increments the count
rather than adding a line. A certificate is **valid** only while its hash
matches the lemma's current hash, so editing a lemma, or the statement of
anything it cites, quietly invalidates its certificates.

### Refutations

A **refutation** (`proofs/refutations.py`) records a rejection by `proofs
verify` or `proofs repair`: one file per rejection, holding the lemma's
pair and hash, the verifier, the model, the date and the verifier's
justification. Rejections inside `proofs run` are not refutations; they
are feedback to that run's reviser. A refutation *counts* while its hash
matches the lemma's current hash. When the lemma changes, the refutation
goes stale but is kept, and its justification is shown to the verifiers
the next time the lemma is verified, so a known objection is not
forgotten. A verification that accepts the lemma deletes its refutations.

### Suspension

A lemma is **suspended** (`proofs/suspension.py`) when any of these holds:

- it has no valid certificate;
- it has a refutation that counts;
- it cites a suspended lemma.

Suspension is recomputed from the files every time the DAG is loaded and
never stored. A suspended lemma is hidden from the planner, the selector
and the prover, and the prover may not cite it; the run warns about each
one once. `verify`, `repair` and `status` still load suspended lemmas,
since verification is how they come back. A lemma whose run stopped
between committing it and certifying it is reported as such; run `proofs
verify` on it to give it its certificate.

### Verify, repair, prune and status

- **`proofs verify PATH lemma_id [--full]`** runs the three verifier checks
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
  proof is written *in place* in the owner's DAG file. This is the one
  command that writes another user's DAG file. A new proof under the same
  statement leaves every dependent's certificate valid. If the reviser
  revised the statement, the lemmas citing it lose their certificates, so
  the repair then climbs the dependents in dependency order, re-verifying
  each one whose certificate no longer matches and re-proving any that are
  rejected. A re-proof leaves the certificate of the proof it replaced in
  your certificate file, so a repair that re-proved anything ends by saying
  how many of your certificates are now stale and that `proofs prune DIR`
  drops them.
- **`proofs prune DIR`** drops the lines of your own certificate file
  whose hashes no longer match. Other users' files are never touched.
- **`proofs status PATH lemma_id`** lists, for the lemma and everything it
  cites, each lemma's current hash and its valid certificates, then
  whether the conjecture is settled. It writes nothing.

## Parallel mode

`--parallel N` runs N proof loops concurrently. Each loop is the same engine
the serial run uses — selector, prover, the three verifiers, reviser, one
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

### The shared possible-lemma list

The loops do not take lemmas from the planner directly. Between them sits a
fixed buffer of **N+5 slots** — the possible-lemma list. The generator is its
only writer: each round it proposes five candidate lemmas
(`agents/parallel/parallel_lemma_generator.md`) and the new ones go into the
empty slots. An id already in the DAG, already in the buffer, already on the
reject list, or currently held in flight by a loop is dropped, so no id is
ever claimed twice. The slots are also the flow control: a full buffer means
the loops are behind, and the generator waits for a free slot instead of
piling up work.

Claiming is atomic. A loop's selector is shown the unclaimed lemmas and its
pick is committed under lock; if another loop took the lemma in the meantime,
the selector is re-run on what remains, so two loops never work on one id. A
claimed lemma keeps its slot until it is settled — proved or given up after
`MAX_PROOF_ATTEMPTS` rounds — or the operator proves it from the menu.

The buffer is not the only route to an id: a reviser that decomposes a hard
lemma picks a fresh id the buffer does not know about, and two loops can
land on the same id (both decompose to it, or one decomposes to it while a
sibling proves the buffer's copy of it). The commit to the DAG checks the id
under the DAG lock and the first proof stands under the id; a colliding
proof is never dropped — it is committed under a fresh non-colliding id
(`P1_2`, `P1_3`, ...), and the rename is recorded on the shared reject list
so the planner can see that the two proofs may state different things.

### The planner and the generator

- **Planner** (`agents/parallel/parallel_planner.md`) sets direction, not
  lemmas: a strategy summary and an ordered list of priorities — which gaps
  in the DAG to close next, and why. It runs once at start, and then whenever
  the DAG changes: a lemma added by *any* loop bumps the DAG version and
  marks the plan stale. If the DAG moves again *during* the planning call,
  the plan is installed already marked stale and the planner goes straight
  back around — no DAG change is lost, and the plan the loops are steering
  by was always written for the DAG as it stood.
- **Generator** (`agents/parallel/parallel_lemma_generator.md`) turns the
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

A parallel run writes a **version-2** checkpoint to the same
`dags/<user_id>_dag.checkpoint.json` beside your DAG file: one slot per loop, each holding that
loop's iteration budget and its in-flight round (or none), plus the shared
reject list. The file is written atomically at the same safe boundaries the
serial run uses. The two checkpoint versions are not interchangeable: a
serial (v1) checkpoint is not a valid parallel one — a parallel run ignores
it and starts fresh from the DAG — and a parallel (v2) checkpoint is not a
valid serial one either, so `--fresh` or deleting the file is the way to get
back to the serial run from a parallel one.

Re-run the same command with the same `--parallel N` to resume: every
loop's budget continues rather than resets, and a loop that was mid-proof
goes back to the prover at the checkpointed round. The checkpoint records how
many loops wrote it, and a run with a different loop count does not load it
(it starts fresh from the DAG instead; the lemmas are kept either way). The
rest of [Cancelling a run (Ctrl-C) and resuming](#cancelling-a-run-ctrl-c-and-resuming)
holds unchanged: Ctrl-C stops every loop and writes the checkpoint first;
a checkpoint without your DAG file beside it is discarded; `--fresh`
restarts the budgets from 1 and keeps the DAG.

## Project layout

```
agents/
  planner.md  selector.md  prover.md  verifier_1/2/3.md  reviser.md   # default agent prompts (one reviser for rejects and overflows)
  parallel/                                    # parallel-mode prompts
    parallel_planner.md  parallel_lemma_generator.md
  parsing.md                               # references parser prompt
proofs/
  cli.py               # the `proofs` command: subcommands and argument validation
  config.py            # the user's global config (~/.config/proofs/config): user.id, user.name, user.email
  main.py              # the proof loop (planner → selector → prover → verifier → reviser) and the verify/repair/prune/status/export entries
  parsing.py           # standalone references/ → references.md parser
  merkle.py            # the Merkle hash of every lemma
  certificates.py      # certificate files: recording, validity, pruning
  refutations.py       # refutation files: recording, loading, dismissal
  suspension.py        # which lemmas are not to be built on, computed from the files
  resolution.py        # whether the conjecture is settled, computed from the files
  export.py            # the LaTeX export
  tools.py             # agent tools (lookup_lemmas) and the two tool-calling protocols
  llm_backend.py       # OpenAI-compatible transport (llama.cpp or a hosted API) + model profiles
  interaction.py       # human-in-the-loop menu + hotkey
  workspace.py         # per-conjecture directories and prompt resolution
conjectures/
  algebra_example/
    conjecture.md          # required: the statement
    dags/
      <user_id>_dag.json             # committed: one user's lemmas
      <user_id>_dag.checkpoint.json  # gitignored: where that user's cancelled run stopped (--fresh)
    certificates/
      <user_id>.jsonl      # committed: the acceptances that user issued
    refutations/           # committed: one file per rejection by verify or repair (created on first write)
    references/            # maintainer-local source files (.tex/.md/plain text), gitignored
    references.md          # committed: parsed theorem-level results (strict JSON array)
    comments.md            # optional, gitignored: this user's private notes for the planner
    prover.md              # optional per-conjecture prompt override
pyproject.toml             # the package and the `proofs` console script
flake.nix                  # Nix dev shell, package and app (`nix run .`)
```

### Conjectures and the shared DAG

`DIR` names a conjecture directory: a name under `conjectures/`, or a path
to one, holding a `conjecture.md`. Every user's lemmas live in their own
`dags/<user_id>_dag.json`, and every command merges all the files in `dags/`
into the complete DAG. Point a second user, or a second model, at a
conjecture already under way and it continues from the lemmas already
proved: a lemma proved by a 27B model and certified is no less proved when
a 122B model arrives. The cost is that this is collaboration, not a
controlled comparison — results are entangled from the first shared lemma.
(DAGs from before the per-user layout — a root `dag.json`, or
`dag-<backend>-<model>.json` files — are no longer read, and the run banner
says so; move one to `dags/<user_id>_dag.json` to carry its lemmas over.
They will be suspended until `proofs verify` certifies them.)

A DAG file is a JSON object: the owner's `user.id`, `user.name` and
`user.email`, then `"lemmas"`, a map from each lemma_id to its node. A node
stores exactly:

| Field | Meaning |
| --- | --- |
| `statement` | The lemma's statement. |
| `proof` | The proof now in the file. |
| `cited_lemmas` | The lemmas the proof cites, as `{"user_id", "lemma_id"}` pairs. Each `user_id` is the cited lemma's **owner**: the name of the file it lives in. |
| `cited_references` | The references the proof cites, by id. |
| `last_prover_id` | Who wrote the proof now in the file. This is the owner for a lemma their own run proved, and the repairer for one that `proofs repair` re-proved in someone else's file. |
| `proved_at` | When that proof was written (the lemma window's recency). |

A lemma's owner is never stored inside the node: it is the file's name,
`<owner>_dag.json`. Its id is the node's key. Only `statement`, `proof` and
the cited statements go into the [Merkle hash](#certificates-and-the-merkle-hash).

Prompt files are looked up in the conjecture directory first, then in
`agents/`, so a conjecture that needs a special prompt (e.g. a prover primed
for Brauer groups) can carry one without forking the defaults.

### Cancelling a run (Ctrl-C) and resuming

Ctrl-C stops the run and leaves a checkpoint —
`dags/<user_id>_dag.checkpoint.json`, beside your DAG file, gitignored. If
Ctrl-C lands while an accepted lemma is being committed and certified, the
run finishes that step first (press Ctrl-C again to force), so a lemma is
never left committed without its certificate. The checkpoint records what a
cancelled run would otherwise lose: where the iteration budget stood, the
planner's reject list (the `failed_attempts` the next planner call reads),
and, if the cancellation landed mid-proof, the lemma in flight, which prover
round it was on, and the feedback the earlier rounds of that attempt already
produced. The file is written at the safe boundaries — the top of each
iteration, before each prover round, and the end of each iteration — so it
always holds a coherent position, and it is deleted when the conjecture
resolves. An existing checkpoint is announced in the run banner.

Re-run the same command to pick up where the run stopped: the budget
continues rather than resets, and an in-flight lemma goes straight back to
the prover at the checkpointed round without asking the planner again. The
interrupted round itself is re-run — the checkpoint cannot reach into an
in-flight LLM call — but everything completed before it is kept. `--fresh`
deletes the checkpoint and restarts the budget from 1 without touching
any DAG file.

A bad checkpoint degrades instead of crashing: an unreadable or incoherent
one is discarded and the run starts fresh; a checkpoint whose in-flight
lemma is already in the DAG drops only the in-flight part and resumes as a
normal iteration; one whose iteration stands past `--max-iterations` is
reported and kept — raise the budget and re-run to continue from it.

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
  `agents/parsing.md`) for the theorem-level results, and writes them to
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
  goes in `cited_lemmas` instead, as a `{"user_id", "lemma_id"}` pair).
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
**Human comments** section, and `agents/planner.md` tells the planner how
to read it: weigh the suggestions — when a comment points at a viable route
the state of the proof allows, one of the five candidates should be a
concrete version of it — but treat them as suggestions, not instructions,
since a comment may be mistaken or stale.

The file is gitignored: each user keeps their own `comments.md`, and
comments are never committed or shared. A clone you pull therefore carries
no one else's notes, and yours never reach anyone else's run — the file is
per user the way the DAG files are, but by absence of sharing rather than
by a `<user_id>` in the name. The planner is the only agent that sees it
(the prover, verifiers and reviser are deliberately left to judge the
mathematics on its own), and a conjecture directory without a `comments.md`
runs with no comments section in the planner's prompt, the same way the
reference blocks degrade. The run banner announces the file when it
is present.
