# Multi-Agent Theorem Prover

A Python program that drives a small team of LLM agents to iteratively prove
(or refute) a mathematical conjecture. The agents work in a loop —
**planner → selector → prover → verifier → reviser** — and accumulate a proof
as a DAG of lemmas written to `dag.json` next to the conjecture. (In
`--mode human`, the operator takes the selector's place at the menu.)

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
   - `--mode human`: a menu appears. Type a number and Enter to accept that
     lemma as true on your own authority (it goes straight into the DAG,
     marked `"provenance": "operator"`). Type `Np` to send a lemma through
     the ordinary proving pipeline instead. You can also write your own
     lemma (`w`, or `wp` to have it proved) or send the planner back.
3. **Prover** (`prover.md`) writes a complete, rigorous proof of the chosen
   lemma, citing only lemmas already in the DAG.
4. **Verifiers** (`verifier_1.md`, `verifier_2.md`, `verifier_3.md`) check it
   adversarially — logic *and* arithmetic. Verification is three atomic
   steps, one per agent; a lemma enters the DAG only after all three have
   accepted it. The three prompts are specialized: `verifier_1.md` checks
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

### Thinking vs. structured output

The prompts instruct each model to emit strict JSON directly, so every
response is first parsed as-is; a schema-constrained *extraction* call only
runs as a fallback. The underlying tension there — on llama.cpp, reasoning
silently voids the grammar — is handled inside `src/llm_backend.py`, which
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

One transport, probed at startup (`src/llm_backend.py`): the
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

Model capabilities (context size, thinking support) are *profiles*, probed
rather than hard-coded, so adding a model needs no code change.

## Setup

A Nix dev shell is provided:

```sh
nix develop
```

It gives you a Python 3 environment with `requests` installed. Otherwise any
Python 3 install with `requests` works.

## Usage

```sh
python src/main.py --conjecture algebra_example
python src/main.py --conjecture algebra_example --model Qwen3.5-122B-Q4_K_M
python src/main.py --conjecture algebra_example --no-verbose --max-iterations 25
python src/main.py --conjecture algebra_example --mode human
python src/main.py --conjecture algebra_example --parallel 2
python src/main.py --conjecture algebra_example --parallel 3 --mode human

# or against a hosted API instead of a local llama-server
LLM_API_KEY=sk-... python src/main.py --conjecture algebra_example \
    --host https://api.openai.com --model gpt-4o

# once, before proving: parse the conjecture's reference collection
python src/parsing.py --conjecture algebra_example
```

`parsing.py` shares main.py's server options (`--host`, `--api-key`,
`--backend`, `--model`, `--num-ctx`) and its `--conjecture NAME` /
`--verbose` pair; it has no proving options, because it has no loop.

### CLI options

| Option | Default | Meaning |
| --- | --- | --- |
| `--conjecture NAME` | `algebra_example` | A directory under `conjectures/` holding a `conjecture.md`. |
| `--model NAME` | `qwen3.8-27b` | Model name: the `--alias` llama-server was launched with, or a provider's model id (e.g. `gpt-4o`). |
| `--host URL` | `http://localhost:8081` | OpenAI-compatible base URL — a local llama-server, or a hosted API's base (e.g. `https://api.openai.com`, without the `/v1` suffix — it is added automatically). `$LLAMA_HOST` overrides the default; 8080 is taken by open-webui. |
| `--api-key KEY` | `$LLM_API_KEY` | Sent as `Authorization: Bearer <key>`. Needed for a hosted API; leave unset for a plain local llama-server. |
| `--backend {auto,llamacpp,openai}` | `auto` | Which dialect of the OpenAI-compatible endpoint to speak: the full llama.cpp option set (with thinking suppressed for schema calls), or the standard OpenAI set. `auto` takes the probe's answer; the other two force it, for when the probe can't see the real server. |
| `--dag PATH` | `conjectures/<name>/dag.json` | Use a separate DAG file (e.g. to isolate a model's run). |
| `--max-iterations N` | 10 | Loop iterations before giving up. |
| `--max-proof-attempts N` | 3 | Prover rounds per lemma before it is given up. |
| `--fresh` | off | Ignore and delete the checkpoint (`<dag>.checkpoint.json`), restarting the iteration budget from 1. The DAG is kept. |
| `--parallel [N]` | off | Run N proof loops in parallel around one shared DAG, steered by one planner and one lemma generator. Bare `--parallel` is N=1. `--mode human`, when set, takes the place of one of the loops (loop-0). See [Parallel mode](#parallel-mode). |
| `--num-ctx N` | the server's own (probed) | Context window in tokens. On a local llama-server: defaults to the `-c` it was launched with, as probed — lower it if VRAM is tight — and an explicit value is clamped to that ceiling. On a hosted `--host` API nothing is probed, so the default is 65536 and an explicit value replaces it. Budgeting above the server's own window is caught at once from the server's 400, which names its real limit and the `--num-ctx` to re-run with. |
| `--mode {auto,human}` | `auto` | Automated selection vs. human menu at the planning step. |
| `--hotkey KEY` | `h` | Keystroke that toggles auto/human mid-run (pass `''` to disable). |
| `--verbose / --no-verbose` | on | Console logging. |

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

`src/interaction.py` explains why the toggle is a keystroke in one place and
a line of input in the other.

## Parallel mode

`--parallel N` runs N proof loops concurrently. Each loop is the same engine
the serial run uses — selector, prover, the three verifiers, reviser, one
lemma at a time — and every loop reads and writes the same DAG. Two further
threads steer them: a **planner** that keeps a shared strategy current, and a
**lemma generator** that keeps the loops supplied with work.

```sh
python src/main.py --conjecture algebra_example --parallel 2
python src/main.py --conjecture algebra_example --parallel 3 --mode human
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
`MAX_PROOF_ATTEMPTS` rounds — or the operator asserts it from the menu.

The buffer is not the only route to an id: a reviser that decomposes a hard
lemma picks a fresh id the buffer does not know about, and two loops can
land on the same id (both decompose to it, or one decomposes to it while a
sibling proves the buffer's copy of it). The commit to the DAG checks the id
under the DAG lock and the first proof stands under the id; a colliding
proof is never dropped — it is committed under a fresh non-colliding id
(`P1_2`, `P1_3`, ...), marked with `"renamed_from": "P1"` so the DAG stays
self-explanatory, and the rename is recorded on the shared reject list so
the planner can see that the two proofs may state different things.

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
serial menu can do works the same: assert a lemma on your authority (it goes
straight into the shared DAG as `"provenance": "operator"`), send one through
the proving pipeline, write your own, or reject the buffer and send the
planner back. The hotkey and the menu's `a` still hand loop-0 between human
and automatic mid-run; the other loops are always automatic. Quitting from
the menu stops the whole run, not just
loop-0 — the other loops' in-flight lemmas go into the checkpoint and resume
next time.

### Checkpointing and resuming

A parallel run writes a **version-2** checkpoint to the same
`<dag>.checkpoint.json` beside the DAG: one slot per loop, each holding that
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
a checkpoint without a `dag.json` is discarded; `--fresh` restarts the
budgets from 1 and keeps the DAG.

## Project layout

```
agents/
  planner.md  selector.md  prover.md  verifier_1/2/3.md  reviser.md   # default agent prompts (one reviser for rejects and overflows)
  parallel/                                    # parallel-mode prompts
    parallel_planner.md  parallel_lemma_generator.md
  parsing.md                               # references parser prompt
src/
  main.py              # the proof loop (planner → selector → prover → verifier → reviser)
  parsing.py           # standalone references/ → references.md parser
  llm_backend.py       # OpenAI-compatible transport (llama.cpp or a hosted API) + model profiles
  interaction.py       # human-in-the-loop menu + hotkey
  workspace.py         # per-conjecture run directories and prompt resolution
conjectures/
  algebra_example/
    conjecture.md          # required: the statement
    dag.json               # shared by every model
    dag.checkpoint.json    # optional: where a cancelled run stopped (--fresh)
    references/            # optional: source files (.tex/.md/plain text) to parse
    references.md          # parsed theorem-level results (strict JSON array)
    comments.md            # optional: operator notes, passed to the planner as suggestions
    prover.md              # optional per-conjecture prompt override
flake.nix                  # Nix dev shell
```

### Conjectures and the shared DAG

`--conjecture` names a directory under `conjectures/` that must contain a
`conjecture.md`. The DAG is written beside it as `dag.json` and is *shared by
every model*: point a second model at a conjecture already under
way and it continues from the lemmas the first one proved. A lemma proved by
a 27B model and passed by the verifiers is no less proved when a 122B model
arrives. The cost is that this is collaboration, not a controlled comparison —
give a model its own file with `--dag` if you want them isolated again.

Prompt files are looked up in the conjecture directory first, then in
`agents/`, so a conjecture that needs a special prompt (e.g. a prover primed
for Brauer groups) can carry one without forking the defaults.

### Cancelling a run (Ctrl-C) and resuming

Ctrl-C stops the run and leaves a checkpoint — `<dag>.checkpoint.json`
beside `dag.json`, so a `--dag` override gets its own. It records what a
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
`dag.json`; the DAG itself still needs deleting for a genuinely clean run.

A bad checkpoint degrades instead of crashing: an unreadable or incoherent
one is discarded and the run starts fresh; a checkpoint whose in-flight
lemma is already in the DAG drops only the in-flight part and resumes as a
normal iteration; one whose iteration stands past `--max-iterations` is
reported and kept — raise the budget and re-run to continue from it.

### References

A conjecture can ship a `references/` directory: chapters, surveys or
papers, as `.tex`, `.md` or plain text, that the agents may lean on. The
directory is parsed by a standalone tool, not by the proving loop:

```sh
python src/parsing.py --conjecture algebra_example
```

reads every parsable file in `references/`, asks the model (via
`agents/parsing.md`) for the theorem-level results, and writes them to
`references.md` beside the DAG as a **strict JSON array** — one object per
result with `id`, `slogan`, `formal statement`, `reference` (where it appears
in the file) and `tags`. `main.py` then feeds that file to the loop at three
granularities:

- **Planner and reviser** — an `id` + `slogan` shortlist, and only while the
  collection is small: below `PLANNER_REFERENCE_LIMIT` (100) entries each
  gets every result, at or above it none. A shortlist of hundreds of
  slogans costs more than it returns, and the prover sees the full
  collection anyway, so a named result the planner misses is one round
  away, not lost. A strategy for large collections is deliberately not
  built yet.
- **Prover** — the whole collection unconditionally (id, slogan, formal
  statement, tags). It may cite any entry by `id` in `cited_lemmas` and
  rely on it without proving it, exactly like a proved lemma.
- **Verifier** — only the formal statements of the results the proof
  actually cites, merged with the cited DAG lemmas into one set. The rest
  of the collection is withheld on purpose: a proof that leans on a result
  it never cited must read as an unjustified leap, and the verifier is
  told that citing a reference is legitimate while proving it inline is
  not.

Two properties of the parser matter for resuming:

- **Ids are stable across re-parses.** `ref_N` numbers are assigned by the
  tool, never by the model; a result whose formal statement matches an
  existing entry keeps its id, so a DAG that cites `ref_3` still means the
  same thing after you add a chapter to `references/` and re-run.
- **Re-parses are a union.** Entries from a previous `references.md` whose
  statements were not re-derived this run are kept, so a flaky parse of one
  file cannot silently delete a result another proof cites. An empty
  `references/` directory never touches an existing `references.md`. There is
  no clear mode: to reset the collection, delete or rewrite `references.md`
  (a DAG that cited its entries will warn about the dangling ids) — deleting
  it and re-running `parsing.py` is the cleanest way to start over.

A run without `references/` or `references.md` is exactly the old run:
every prompt degrades to its pre-references form, which is also the failure
mode for a corrupt `references.md` (it warns once and is ignored).

### Comments (operator hints to the planner)

A conjecture can ship a `comments.md` beside its `conjecture.md`: free-form
notes on possible approaches to a proof or a counterexample — routes to try,
objects or theorems worth using, or dead ends to avoid. No tooling touches
it; you just edit it. On every planning step the loop hands the file's
contents to the planner verbatim as a **Human comments** section, and
`agents/planner.md` tells the planner how to read it: weigh the suggestions
— when a comment points at a viable route the state of the proof allows,
one of the five candidates should be a concrete version of it — but treat
them as suggestions, not instructions, since a comment may be mistaken or
stale.

The planner is the only agent that sees the file (the prover, verifiers and
reviser are deliberately left to judge the mathematics on its own), and a
conjecture directory without a `comments.md` runs exactly as before — the
prompt degrades to its pre-comments form, the same way the reference blocks
do. The run banner announces the file when it is present.
